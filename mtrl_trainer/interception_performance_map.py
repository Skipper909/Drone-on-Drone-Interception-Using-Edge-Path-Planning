#!/usr/bin/env python3
"""
interception_performance_map.py

Run many randomized interception episodes and analyze performance vs initial
relative geometry (p_rel). Designed to run with NO CLI args:

  python3 interception_performance_map.py

Thesis-oriented outputs (PNG + PDF):
  - outcome_stack.(png|pdf)                 success/near-miss/fail rates
  - success_vs_range.(png|pdf)              binned success vs range with 95% Wilson CI + counts
  - success_vs_bearing.(png|pdf)            binned success vs bearing with 95% Wilson CI + counts
  - hexbin_success_xy.(png|pdf)             2D hexbin success-rate vs (x,y) with mincnt filtering
  - hexbin_min_dist_xy.(png|pdf)            2D hexbin mean min distance vs (x,y) with mincnt filtering
  - first_pass_vs_abs_y.(png|pdf)           binned first-pass rate vs |y| with 95% Wilson CI + counts

Path resolution (eval_stats-style):
  - _CURRENT_DIR = directory of this file (mtrl_trainer/)
  - _PROJECT_ROOT = parent directory (repo root)
  - resolve_path() tries: as-given, relative to _CURRENT_DIR, relative to _PROJECT_ROOT

Notes:
  - Default max_steps ~ 25s/dt = 1300
  - Early stop on capture (Python-side) is enabled by default
  - Uses p_rel from task_specific_obs[0:3] by default (your current usage)
"""

import os
import csv
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.environments import tf_py_environment

try:
    from tf_agents.policies import greedy_policy
except Exception:
    greedy_policy = None

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file


# -----------------------------
# Path resolution (eval_stats style)
# -----------------------------

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))


def resolve_path(p: str) -> str:
    """Resolve path: (1) as-given, (2) relative to _CURRENT_DIR, (3) relative to _PROJECT_ROOT."""
    if p is None:
        return p
    p = str(p)

    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)

    p2 = os.path.join(_CURRENT_DIR, p)
    if os.path.exists(p2):
        return os.path.abspath(p2)

    p3 = os.path.join(_PROJECT_ROOT, p)
    if os.path.exists(p3):
        return os.path.abspath(p3)

    return os.path.abspath(p)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_slice(s: str) -> slice:
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid slice '{s}', expected 'a:b' or 'a:b:c'")
    vals = []
    for p in parts:
        p = p.strip()
        vals.append(None if p == "" else int(p))
    if len(vals) == 2:
        return slice(vals[0], vals[1], None)
    return slice(vals[0], vals[1], vals[2])


def to_np(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def is_last_ts(time_step) -> bool:
    v = time_step.is_last()
    if isinstance(v, tf.Tensor):
        v = v.numpy()
    v = np.asarray(v)
    if v.ndim == 0:
        return bool(v)
    return bool(v[0])


# -----------------------------
# Policy handling (eval_stats style)
# -----------------------------

def load_policy(policy_dir: str):
    if not tf.io.gfile.exists(policy_dir):
        raise FileNotFoundError(
            f"[eval] POLICY_DIR does not exist: {policy_dir}\n"
            "Set DEFAULT_POLICY_DIR in the script or pass --policy_dir."
        )

    loaded = tf.saved_model.load(policy_dir)
    base = getattr(loaded, "policy", loaded)

    # Try to force deterministic if it looks like a TF-Agents policy
    if greedy_policy is not None and hasattr(base, "time_step_spec"):
        try:
            print("[eval] Wrapping policy with TF-Agents GreedyPolicy (deterministic).")
            return greedy_policy.GreedyPolicy(base)
        except Exception as e:
            print(f"[eval] GreedyPolicy wrap failed ({type(e).__name__}: {e}). Using base policy directly.")

    print("[eval] Using base SavedModel policy directly (may be stochastic).")
    return base


def get_initial_state(policy, batch_size: int):
    if hasattr(policy, "get_initial_state"):
        try:
            return policy.get_initial_state(batch_size)
        except Exception:
            pass
    return ()


def policy_action(policy, time_step, policy_state):
    try:
        return policy.action(time_step, policy_state)
    except TypeError:
        return policy.action(time_step)


# -----------------------------
# Environment build (eval_stats style)
# -----------------------------

def build_env(sim_config_path: str,
              agi_param_dir: str,
              sim_base_dir: str,
              num_drones: int = 1,
              deterministic_evalmode: bool = False):
    track_json = os.path.join(sim_base_dir, "track.json")
    if not os.path.exists(track_json):
        raise FileNotFoundError(f"[eval] track.json not found at: {track_json}")

    track_layout = load_track_from_file(track_json)

    py_env = BatchedAgiSimEnv(
        sim_config_path=sim_config_path,
        agi_param_dir=agi_param_dir,
        sim_base_dir=sim_base_dir,
        num_drones=num_drones,
        track_layout=track_layout,
    )

    if deterministic_evalmode and hasattr(py_env, "setEval"):
        print("[eval] setEval() enabled => deterministic target reset (evalMode=true)")
        py_env.setEval()
    else:
        print("[eval] setEval() NOT called => training randomization active (evalMode=false)")

    eval_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"[eval] Environment batch size: {eval_env.batch_size}")
    return eval_env


# -----------------------------
# Episode record
# -----------------------------

@dataclass
class EpisodeRow:
    episode: int
    success: int
    near_miss: int
    fail: int
    first_pass_success: int
    steps: int
    time_s: float
    capture_time_s: float  # nan if not captured
    min_dist_m: float
    init_px_m: float
    init_py_m: float
    init_pz_m: float
    init_dist_m: float
    init_bearing_rad: float


# -----------------------------
# Observation extraction
# -----------------------------

def _get_obs_vec(obs: Dict[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"obs has no '{key}'")
    v = to_np(obs[key])
    v = np.asarray(v)
    if v.ndim == 2:
        v = v[0]
    return v


def extract_p_rel_scaled(obs: Dict[str, Any], source: str, sl: slice) -> np.ndarray:
    if source == "task":
        v = _get_obs_vec(obs, "task_specific_obs")
        return np.asarray(v[sl], dtype=np.float32)
    if source == "shared":
        v = _get_obs_vec(obs, "shared_obs")
        return np.asarray(v[sl], dtype=np.float32)
    if source == "auto":
        if "task_specific_obs" in obs:
            v = _get_obs_vec(obs, "task_specific_obs")
            if v.size >= (sl.stop or 0):
                pr = np.asarray(v[sl], dtype=np.float32)
                if pr.size >= 3:
                    return pr
        if "shared_obs" in obs:
            v = _get_obs_vec(obs, "shared_obs")
            if v.size >= (sl.stop or 0):
                pr = np.asarray(v[sl], dtype=np.float32)
                if pr.size >= 3:
                    return pr
        raise RuntimeError("auto p_rel extraction failed; set --p_rel_source and --p_rel_slice explicitly")
    raise ValueError(f"Unknown p_rel source '{source}'")


# -----------------------------
# Episode runner
# -----------------------------

def run_episode(eval_env,
                policy,
                max_steps: int,
                p_rel_source: str,
                p_rel_slice: slice,
                rel_pos_scale_m: float,
                capture_radius_m: float,
                near_miss_radius_m: float,
                dt: float,
                stop_on_capture: bool) -> EpisodeRow:
    ts = eval_env.reset()
    policy_state = get_initial_state(policy, eval_env.batch_size)

    p_rel0_scaled = extract_p_rel_scaled(ts.observation, p_rel_source, p_rel_slice)[:3]
    p_rel0_m = p_rel0_scaled * float(rel_pos_scale_m)

    init_dist_m = float(np.linalg.norm(p_rel0_m))
    init_bearing = float(np.arctan2(p_rel0_m[1], p_rel0_m[0]))

    # Unit direction for first-pass logic
    p_hat = p_rel0_m / (init_dist_m + 1e-9)

    min_dist_m = init_dist_m
    passed_target = False

    capture_step = None  # first step at which within capture radius

    step_count = 0
    while True:
        p_rel_scaled = extract_p_rel_scaled(ts.observation, p_rel_source, p_rel_slice)[:3]
        p_rel_m = p_rel_scaled * float(rel_pos_scale_m)

        dist_m = float(np.linalg.norm(p_rel_m))
        if dist_m < min_dist_m:
            min_dist_m = dist_m

        if (capture_step is None) and (min_dist_m <= float(capture_radius_m)):
            capture_step = step_count

        if float(np.dot(p_rel_m, p_hat)) < 0.0:
            passed_target = True

        if stop_on_capture and (min_dist_m <= float(capture_radius_m)):
            break

        if is_last_ts(ts) or step_count >= max_steps - 1:
            break

        a = policy_action(policy, ts, policy_state)
        if hasattr(a, "state"):
            policy_state = a.state

        ts = eval_env.step(a.action)
        step_count += 1

    success = int(min_dist_m <= float(capture_radius_m))
    near_miss = int((min_dist_m > float(capture_radius_m)) and (min_dist_m <= float(near_miss_radius_m)))
    fail = int(min_dist_m > float(near_miss_radius_m))

    first_pass_success = int(success and (not passed_target))

    steps_used = int(step_count)
    time_s = float((steps_used + 1) * float(dt))
    capture_time_s = float((capture_step + 1) * float(dt)) if (capture_step is not None) else float("nan")

    return EpisodeRow(
        episode=-1,
        success=success,
        near_miss=near_miss,
        fail=fail,
        first_pass_success=first_pass_success,
        steps=steps_used,
        time_s=time_s,
        capture_time_s=capture_time_s,
        min_dist_m=float(min_dist_m),
        init_px_m=float(p_rel0_m[0]),
        init_py_m=float(p_rel0_m[1]),
        init_pz_m=float(p_rel0_m[2]),
        init_dist_m=float(init_dist_m),
        init_bearing_rad=float(init_bearing),
    )


# -----------------------------
# Stats helpers
# -----------------------------

def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wilson score interval for binomial proportion.
    Returns (p_hat, lo, hi). For n==0 returns nan.
    """
    k = k.astype(float)
    n = n.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = k / n
        denom = 1.0 + (z**2) / n
        center = (phat + (z**2) / (2.0 * n)) / denom
        margin = (z * np.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * n)) / n)) / denom
        lo = center - margin
        hi = center + margin
    phat[n == 0] = np.nan
    lo[n == 0] = np.nan
    hi[n == 0] = np.nan
    return phat, lo, hi


def binned_rate_with_ci(x: np.ndarray, y01: np.ndarray, edges: np.ndarray):
    """
    Bin x into edges; compute n, k, rate and Wilson CI for y01 (0/1).
    Returns: centers, rate, lo, hi, n
    """
    x = np.asarray(x)
    y01 = np.asarray(y01).astype(int)
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    n = np.zeros(nb, dtype=int)
    k = np.zeros(nb, dtype=int)
    for i in range(nb):
        m = idx == i
        n[i] = int(np.sum(m))
        k[i] = int(np.sum(y01[m])) if n[i] > 0 else 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate, lo, hi = wilson_ci(k, n)
    return centers, rate, lo, hi, n


def save_csv(rows, path):
    fields = [
        "episode",
        "success", "near_miss", "fail",
        "first_pass_success",
        "steps", "time_s", "capture_time_s",
        "min_dist_m",
        "init_px_m", "init_py_m", "init_pz_m",
        "init_dist_m", "init_bearing_rad",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def save_fig(path_png: str, path_pdf: str):
    plt.tight_layout()
    plt.savefig(path_png, dpi=250)
    plt.savefig(path_pdf)
    plt.close()


# -----------------------------
# Thesis plots
# -----------------------------

def plot_outcome_stack(success_rate: float, near_miss_rate: float, fail_rate: float, out_dir: str):
    plt.figure(figsize=(7.5, 2.8))
    vals = [success_rate, near_miss_rate, fail_rate]
    labels = ["Success (≤ capture)", "Near-miss (capture–near)", "Fail (> near)"]

    left = 0.0
    for v, lab in zip(vals, labels):
        plt.barh([0], [v], left=[left], label=lab)
        left += v

    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel("Rate")
    plt.title("Interception outcomes (aggregated)")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=1, frameon=False)

    save_fig(
        os.path.join(out_dir, "outcome_stack.png"),
        os.path.join(out_dir, "outcome_stack.pdf"),
    )


def plot_rate_vs_x_with_counts(x, y01, edges, xlabel, title, out_base, out_dir):
    centers, rate, lo, hi, n = binned_rate_with_ci(x, y01, edges)

    fig = plt.figure(figsize=(8.8, 4.6))
    ax = plt.gca()
    ax.plot(centers, rate, marker="o")
    ax.fill_between(centers, lo, hi, alpha=0.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate")
    ax.set_title(title)

    # counts on secondary axis
    ax2 = ax.twinx()
    ax2.plot(centers, n, linestyle="--", marker="x")
    ax2.set_ylabel("Bin count")

    save_fig(
        os.path.join(out_dir, f"{out_base}.png"),
        os.path.join(out_dir, f"{out_base}.pdf"),
    )


def plot_hexbin(x, y, c, reduce_fn, gridsize, extent, mincnt, title, clabel, out_base, out_dir, vmin=None, vmax=None):
    plt.figure(figsize=(8.2, 6.2))
    hb = plt.hexbin(
        x, y, C=c,
        reduce_C_function=reduce_fn,
        gridsize=gridsize,
        extent=extent,
        mincnt=mincnt
    )
    if vmin is not None or vmax is not None:
        hb.set_clim(vmin=vmin, vmax=vmax)
    cb = plt.colorbar()
    cb.set_label(clabel)

    plt.xlabel("Initial p_rel.x [m]")
    plt.ylabel("Initial p_rel.y [m]")
    plt.title(title)
    plt.axis("equal")

    save_fig(
        os.path.join(out_dir, f"{out_base}.png"),
        os.path.join(out_dir, f"{out_base}.pdf"),
    )


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_POLICY_DIR = "policies_intercept/best_intercept_10msTarget_copy"
DEFAULT_SIM_CONFIG = "mtrl_trainer/parameters/simulation.yaml"
DEFAULT_AGI_PARAM_DIR = "agilib/params"
DEFAULT_SIM_BASE_DIR = "mtrl_trainer/parameters"
DEFAULT_OUTPUT_DIR = "perf_maps"

DEFAULT_REL_POS_SCALE_M = 300.0
DEFAULT_DT = 0.02
DEFAULT_MAX_STEPS = 1300          # ~25s / 0.02

# For *thesis* quality, you will want >1000 episodes.
# Keep default moderate so `python3 interception_performance_map.py` completes reasonably.
DEFAULT_NUM_EPISODES = 1000

DEFAULT_CAPTURE_RADIUS = 1.0
DEFAULT_NEAR_MISS_RADIUS = 1.5

# For hexbins: require some support per bin to avoid misleading “salt and pepper”
DEFAULT_HEX_MINCNT = 5
DEFAULT_HEX_GRIDSIZE = 25


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--policy_dir", default=DEFAULT_POLICY_DIR)
    ap.add_argument("--sim_config", default=DEFAULT_SIM_CONFIG)
    ap.add_argument("--agi_param_dir", default=DEFAULT_AGI_PARAM_DIR)
    ap.add_argument("--sim_base_dir", default=DEFAULT_SIM_BASE_DIR)

    ap.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)

    ap.add_argument("--p_rel_source", choices=["task", "shared", "auto"], default="task")
    ap.add_argument("--p_rel_slice", default="0:3")

    ap.add_argument("--rel_pos_scale_m", type=float, default=DEFAULT_REL_POS_SCALE_M)
    ap.add_argument("--capture_radius", type=float, default=DEFAULT_CAPTURE_RADIUS)
    ap.add_argument("--near_miss_radius", type=float, default=DEFAULT_NEAR_MISS_RADIUS)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)

    # stop_on_capture defaults to True, can be disabled with --no_stop_on_capture
    ap.add_argument("--no_stop_on_capture", action="store_false", dest="stop_on_capture",
                    help="Disable early stop on capture.")
    ap.set_defaults(stop_on_capture=True)

    ap.add_argument("--hex_gridsize", type=int, default=DEFAULT_HEX_GRIDSIZE)
    ap.add_argument("--hex_mincnt", type=int, default=DEFAULT_HEX_MINCNT)

    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic_evalmode", action="store_true")

    ap.add_argument("--print_every", type=int, default=50,
                    help="Print progress every N episodes.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    policy_dir = resolve_path(args.policy_dir)
    sim_config = resolve_path(args.sim_config)
    agi_param_dir = resolve_path(args.agi_param_dir)
    sim_base_dir = resolve_path(args.sim_base_dir)
    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)

    print("[paths] _CURRENT_DIR  =", _CURRENT_DIR)
    print("[paths] _PROJECT_ROOT =", _PROJECT_ROOT)
    print("[paths] policy_dir    =", policy_dir)
    print("[paths] sim_config    =", sim_config)
    print("[paths] agi_param_dir =", agi_param_dir)
    print("[paths] sim_base_dir  =", sim_base_dir)
    print("[paths] output_dir    =", out_dir)
    print(f"[run] num_episodes={args.num_episodes} max_steps={args.max_steps} "
          f"stop_on_capture={args.stop_on_capture} capture={args.capture_radius} near={args.near_miss_radius}")

    meta = {
        "policy_dir": policy_dir,
        "sim_config": sim_config,
        "agi_param_dir": agi_param_dir,
        "sim_base_dir": sim_base_dir,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "p_rel_source": args.p_rel_source,
        "p_rel_slice": args.p_rel_slice,
        "rel_pos_scale_m": args.rel_pos_scale_m,
        "capture_radius": args.capture_radius,
        "near_miss_radius": args.near_miss_radius,
        "dt": args.dt,
        "stop_on_capture": bool(args.stop_on_capture),
        "hex_gridsize": args.hex_gridsize,
        "hex_mincnt": args.hex_mincnt,
        "seed": args.seed,
        "deterministic_evalmode": bool(args.deterministic_evalmode),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    p_rel_sl = parse_slice(args.p_rel_slice)

    print(f"[eval] Loading policy from: {policy_dir}")
    policy = load_policy(policy_dir)

    print("[eval] Building environment...")
    env = build_env(
        sim_config_path=sim_config,
        agi_param_dir=agi_param_dir,
        sim_base_dir=sim_base_dir,
        num_drones=1,
        deterministic_evalmode=args.deterministic_evalmode,
    )

    rows = []
    t0 = time.time()
    for ep in range(args.num_episodes):
        row = run_episode(
            eval_env=env,
            policy=policy,
            max_steps=args.max_steps,
            p_rel_source=args.p_rel_source,
            p_rel_slice=p_rel_sl,
            rel_pos_scale_m=args.rel_pos_scale_m,
            capture_radius_m=args.capture_radius,
            near_miss_radius_m=args.near_miss_radius,
            dt=args.dt,
            stop_on_capture=args.stop_on_capture,
        )
        row.episode = ep
        rows.append(row)

        if args.print_every > 0 and ((ep + 1) % args.print_every == 0):
            succ = 100.0 * float(np.mean([r.success for r in rows]))
            nm = 100.0 * float(np.mean([r.near_miss for r in rows]))
            fps = 100.0 * float(np.mean([r.first_pass_success for r in rows]))
            print(f"[{ep+1:>5}/{args.num_episodes}] success={succ:5.1f}% near={nm:5.1f}% first-pass={fps:5.1f}%")

    elapsed = time.time() - t0
    print(f"Finished {args.num_episodes} episodes in {elapsed:.1f}s ({elapsed/max(1,args.num_episodes):.2f}s/ep)")

    csv_path = os.path.join(out_dir, "episodes.csv")
    save_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    # Aggregate arrays
    init_px = np.array([r.init_px_m for r in rows], dtype=np.float32)
    init_py = np.array([r.init_py_m for r in rows], dtype=np.float32)
    init_dist = np.array([r.init_dist_m for r in rows], dtype=np.float32)
    init_bearing = np.array([r.init_bearing_rad for r in rows], dtype=np.float32)

    success = np.array([r.success for r in rows], dtype=np.int32)
    near_miss = np.array([r.near_miss for r in rows], dtype=np.int32)
    fail = np.array([r.fail for r in rows], dtype=np.int32)
    first_pass = np.array([r.first_pass_success for r in rows], dtype=np.int32)
    min_dist = np.array([r.min_dist_m for r in rows], dtype=np.float32)

    # Summary text
    succ_rate = float(np.mean(success))
    nm_rate = float(np.mean(near_miss))
    fail_rate = float(np.mean(fail))
    fp_rate = float(np.mean(first_pass))

    cap_times = np.array([r.capture_time_s for r in rows], dtype=np.float32)
    cap_times_ok = cap_times[np.isfinite(cap_times)]

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"episodes: {args.num_episodes}\n")
        f.write(f"success_rate (<= {args.capture_radius:.2f} m): {succ_rate*100.0:.2f}%\n")
        f.write(f"near_miss_rate (<= {args.near_miss_radius:.2f} m): {nm_rate*100.0:.2f}%\n")
        f.write(f"fail_rate: {fail_rate*100.0:.2f}%\n")
        f.write(f"first_pass_rate: {fp_rate*100.0:.2f}%\n")
        f.write(f"mean_min_dist: {float(np.mean(min_dist)):.3f} m\n")
        if cap_times_ok.size > 0:
            f.write(f"mean_capture_time_success: {float(np.mean(cap_times_ok)):.3f} s\n")
        f.write(f"dt: {args.dt:.4f}\n")
        f.write(f"max_steps: {args.max_steps}\n")
        f.write(f"rel_pos_scale_m: {args.rel_pos_scale_m:.3f}\n")
        f.write(f"hex_gridsize: {args.hex_gridsize}\n")
        f.write(f"hex_mincnt: {args.hex_mincnt}\n")
    print(f"Wrote {summary_path}")

    # -----------------------------
    # Thesis plots
    # -----------------------------

    plot_outcome_stack(succ_rate, nm_rate, fail_rate, out_dir)

    # 1D: success vs range
    r_edges = np.linspace(np.nanpercentile(init_dist, 1), np.nanpercentile(init_dist, 99), 16)
    plot_rate_vs_x_with_counts(
        init_dist, success,
        r_edges,
        xlabel="Initial range ||p_rel|| [m]",
        title="Success rate vs initial range (Wilson 95% CI)",
        out_base="success_vs_range",
        out_dir=out_dir
    )

    # 1D: success vs bearing
    # bearing in [-pi, pi]
    b_edges = np.linspace(-np.pi, np.pi, 17)
    plot_rate_vs_x_with_counts(
        init_bearing, success,
        b_edges,
        xlabel="Initial bearing atan2(y, x) [rad]",
        title="Success rate vs initial bearing (Wilson 95% CI)",
        out_base="success_vs_bearing",
        out_dir=out_dir
    )

    # 1D: first-pass vs |y|
    abs_y = np.abs(init_py)
    y_edges = np.linspace(0.0, np.nanpercentile(abs_y, 99), 16)
    plot_rate_vs_x_with_counts(
        abs_y, first_pass,
        y_edges,
        xlabel="Initial lateral offset |p_rel.y| [m]",
        title="First-pass capture rate vs lateral offset (Wilson 95% CI)",
        out_base="first_pass_vs_abs_y",
        out_dir=out_dir
    )

    # 2D: hexbin success rate vs (x, y)
    # Set extent from robust percentiles to avoid a few outliers dominating.
    xmin, xmax = np.nanpercentile(init_px, [2, 98])
    ymin, ymax = np.nanpercentile(init_py, [2, 98])
    extent = [float(xmin), float(xmax), float(ymin), float(ymax)]

    plot_hexbin(
        init_px, init_py, success.astype(float),
        reduce_fn=np.mean,
        gridsize=args.hex_gridsize,
        extent=extent,
        mincnt=args.hex_mincnt,
        title=f"Success rate vs initial (p_rel.x, p_rel.y)\n(mincnt={args.hex_mincnt}, gridsize={args.hex_gridsize})",
        clabel="Success rate",
        out_base="hexbin_success_xy",
        out_dir=out_dir,
        vmin=0.0, vmax=1.0
    )

    plot_hexbin(
        init_px, init_py, min_dist.astype(float),
        reduce_fn=np.mean,
        gridsize=args.hex_gridsize,
        extent=extent,
        mincnt=args.hex_mincnt,
        title=f"Mean minimum distance vs initial (p_rel.x, p_rel.y)\n(mincnt={args.hex_mincnt}, gridsize={args.hex_gridsize})",
        clabel="Mean min distance [m]",
        out_base="hexbin_min_dist_xy",
        out_dir=out_dir
    )

    print(f"All thesis plots written to: {out_dir}")


if __name__ == "__main__":
    main()
