#!/usr/bin/env python3
"""
interception_stratified_eval.py

Stratified interception performance evaluation.

Episodes are stratified by:
  1) Initial range ||p_rel||  (m)
  2) Target speed proxy ||v_rel|| (m/s)  [robust finite-difference fallback]
  3) Encounter geometry:
       - Initial bearing atan2(y, x) (rad)
       - Approach angle between v_rel and -p_rel (rad)
       - Lateral ratio |y|/||p_rel|| (unitless)

Outputs (in OUTPUT_DIR):
  - episodes.csv                      per-episode raw metrics
  - bins_range.csv                    per-range-bin metrics
  - bins_speed.csv                    per-speed-bin metrics
  - bins_bearing.csv                  per-bearing-bin metrics
  - bins_approach_angle.csv           per-approach-angle-bin metrics
  - bins_lateral_ratio.csv            per-lateral-ratio-bin metrics
  - report.md                         thesis-ready summary + tables
  - plots/*.png and plots/*.pdf       thesis-friendly plots

Runs with no arguments:
  python3 interception_stratified_eval.py

Path resolution matches eval_stats-style usage:
  - relative to mtrl_trainer/ and to repo root
"""

import os
import csv
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

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

    # Prefer deterministic if possible
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


def extract_vec_scaled(
    obs: Dict[str, Any],
    source: str,
    sl: slice,
    scale: float,
) -> np.ndarray:
    """
    Extract vector from obs using source and slice, then apply scale.
    source in {"task", "shared"}.
    """
    if source == "task":
        v = _get_obs_vec(obs, "task_specific_obs")
    elif source == "shared":
        v = _get_obs_vec(obs, "shared_obs")
    else:
        raise ValueError(f"Unknown source '{source}', expected 'task' or 'shared'")
    out = np.asarray(v[sl], dtype=np.float32) * float(scale)
    return out


def safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)


# -----------------------------
# Episode record
# -----------------------------

@dataclass
class EpisodeRow:
    episode: int

    # outcomes
    success: int
    near_miss: int
    fail: int
    first_pass_success: int

    # timing/quality
    steps: int
    time_s: float
    capture_time_s: float  # nan if not captured
    min_dist_m: float

    # initial condition features
    init_px_m: float
    init_py_m: float
    init_pz_m: float
    init_range_m: float
    init_bearing_rad: float

    # speed/geometry features (computed)
    init_vrel_norm_mps: float
    init_closing_speed_mps: float
    init_approach_angle_rad: float
    init_lateral_ratio: float


# -----------------------------
# Bin stats + CI
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


def stratify_1d(
    x: np.ndarray,
    edges: np.ndarray,
    success: np.ndarray,
    near_miss: np.ndarray,
    first_pass: np.ndarray,
    min_dist: np.ndarray,
    capture_time: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute per-bin metrics for a 1D stratification.
    Returns dict of columns.
    """
    x = np.asarray(x)
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1

    n = np.zeros(nb, dtype=int)
    k_succ = np.zeros(nb, dtype=int)
    k_nm = np.zeros(nb, dtype=int)
    k_fp = np.zeros(nb, dtype=int)

    mean_min = np.full(nb, np.nan, dtype=float)
    mean_cap_t = np.full(nb, np.nan, dtype=float)

    for i in range(nb):
        m = idx == i
        n[i] = int(np.sum(m))
        if n[i] == 0:
            continue
        k_succ[i] = int(np.sum(success[m]))
        k_nm[i] = int(np.sum(near_miss[m]))
        k_fp[i] = int(np.sum(first_pass[m]))
        mean_min[i] = float(np.mean(min_dist[m]))

        # capture time only for successes
        ct = capture_time[m]
        ct = ct[np.isfinite(ct)]
        mean_cap_t[i] = float(np.mean(ct)) if ct.size > 0 else np.nan

    centers = 0.5 * (edges[:-1] + edges[1:])
    succ_rate, succ_lo, succ_hi = wilson_ci(k_succ, n)
    fp_rate, fp_lo, fp_hi = wilson_ci(k_fp, n)

    nm_rate = np.full(nb, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nm_rate = k_nm.astype(float) / n.astype(float)
    nm_rate[n == 0] = np.nan

    fail_rate = np.full(nb, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fail_rate = 1.0 - (k_succ.astype(float) + k_nm.astype(float)) / n.astype(float)
    fail_rate[n == 0] = np.nan

    return {
        "bin_center": centers,
        "bin_lo": edges[:-1],
        "bin_hi": edges[1:],
        "count": n,
        "success_rate": succ_rate,
        "success_ci_lo": succ_lo,
        "success_ci_hi": succ_hi,
        "near_miss_rate": nm_rate,
        "fail_rate": fail_rate,
        "first_pass_rate": fp_rate,
        "first_pass_ci_lo": fp_lo,
        "first_pass_ci_hi": fp_hi,
        "mean_min_dist_m": mean_min,
        "mean_capture_time_s": mean_cap_t,
    }


def write_strat_csv(path: str, cols: Dict[str, np.ndarray]) -> None:
    keys = list(cols.keys())
    nrows = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(nrows):
            w.writerow([cols[k][i] for k in keys])


def plot_rate_with_ci(cols: Dict[str, np.ndarray], xlabel: str, title: str, out_png: str, out_pdf: str) -> None:
    x = cols["bin_center"]
    rate = cols["success_rate"]
    lo = cols["success_ci_lo"]
    hi = cols["success_ci_hi"]
    n = cols["count"]

    fig = plt.figure(figsize=(8.8, 4.6))
    ax = plt.gca()
    ax.plot(x, rate, marker="o")
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate (≤ capture)")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(x, n, linestyle="--", marker="x")
    ax2.set_ylabel("Bin count")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.savefig(out_pdf)
    plt.close()


def plot_stack_overall(success_rate: float, near_miss_rate: float, fail_rate: float, out_png: str, out_pdf: str) -> None:
    plt.figure(figsize=(7.8, 2.8))
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

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Episode runner
# -----------------------------

def run_episode(
    eval_env,
    policy,
    max_steps: int,
    dt: float,
    stop_on_capture: bool,
    rel_pos_source: str,
    rel_pos_slice: slice,
    rel_pos_scale_m: float,
    capture_radius_m: float,
    near_miss_radius_m: float,
    vrel_source: Optional[str],
    vrel_slice: Optional[slice],
    vrel_scale: float,
) -> EpisodeRow:
    """
    vrel extraction logic:
      - if vrel_source and vrel_slice provided: extract v_rel directly from obs and scale by vrel_scale
      - else: estimate v_rel from first-step finite difference of p_rel
    """
    ts0 = eval_env.reset()
    policy_state = get_initial_state(policy, eval_env.batch_size)

    p0 = extract_vec_scaled(ts0.observation, rel_pos_source, rel_pos_slice, rel_pos_scale_m)[:3]
    r0 = float(np.linalg.norm(p0))
    bearing0 = float(np.arctan2(p0[1], p0[0]))
    lateral_ratio0 = float(abs(p0[1]) / (r0 + 1e-9))

    # Obtain v_rel at start
    if vrel_source is not None and vrel_slice is not None:
        v0 = extract_vec_scaled(ts0.observation, vrel_source, vrel_slice, vrel_scale)[:3]
        vrel_is_fd = False
    else:
        # finite difference: step once with policy action
        a0 = policy_action(policy, ts0, policy_state)
        if hasattr(a0, "state"):
            policy_state = a0.state
        ts1 = eval_env.step(a0.action)
        p1 = extract_vec_scaled(ts1.observation, rel_pos_source, rel_pos_slice, rel_pos_scale_m)[:3]
        v0 = (p1 - p0) / float(dt)
        ts0 = ts1  # continue from ts1 to avoid throwing away the step
        vrel_is_fd = True

    vnorm0 = float(np.linalg.norm(v0))
    p_hat = safe_unit(p0)
    v_hat = safe_unit(v0)
    closing_speed0 = float(-np.dot(v0, p_hat))  # positive means closing (range decreasing)
    # approach angle between v_rel and -p_rel
    approach_cos = float(np.clip(np.dot(v_hat, -p_hat), -1.0, 1.0))
    approach_angle0 = float(np.arccos(approach_cos))

    # Rollout for outcome metrics
    min_dist = r0
    passed_target = False
    capture_step = None

    ts = ts0
    step = 0
    while True:
        p = extract_vec_scaled(ts.observation, rel_pos_source, rel_pos_slice, rel_pos_scale_m)[:3]
        dist = float(np.linalg.norm(p))
        if dist < min_dist:
            min_dist = dist

        if capture_step is None and (min_dist <= float(capture_radius_m)):
            capture_step = step

        if float(np.dot(p, p_hat)) < 0.0:
            passed_target = True

        if stop_on_capture and (min_dist <= float(capture_radius_m)):
            break

        if is_last_ts(ts) or step >= max_steps - 1:
            break

        a = policy_action(policy, ts, policy_state)
        if hasattr(a, "state"):
            policy_state = a.state
        ts = eval_env.step(a.action)
        step += 1

    success = int(min_dist <= float(capture_radius_m))
    near_miss = int((min_dist > float(capture_radius_m)) and (min_dist <= float(near_miss_radius_m)))
    fail = int(min_dist > float(near_miss_radius_m))
    first_pass = int(success and (not passed_target))

    steps_used = int(step)
    time_s = float((steps_used + 1) * float(dt))
    capture_time_s = float((capture_step + 1) * float(dt)) if capture_step is not None else float("nan")

    if vrel_is_fd:
        # Slightly annotate by making sure user can diagnose the proxy method.
        pass

    return EpisodeRow(
        episode=-1,
        success=success,
        near_miss=near_miss,
        fail=fail,
        first_pass_success=first_pass,
        steps=steps_used,
        time_s=time_s,
        capture_time_s=capture_time_s,
        min_dist_m=float(min_dist),
        init_px_m=float(p0[0]),
        init_py_m=float(p0[1]),
        init_pz_m=float(p0[2]),
        init_range_m=float(r0),
        init_bearing_rad=float(bearing0),
        init_vrel_norm_mps=float(vnorm0),
        init_closing_speed_mps=float(closing_speed0),
        init_approach_angle_rad=float(approach_angle0),
        init_lateral_ratio=float(lateral_ratio0),
    )


# -----------------------------
# Defaults (edit here to match your repo)
# -----------------------------

DEFAULT_POLICY_DIR = "policies_intercept/best_intercept_10msTarget_copy"
DEFAULT_SIM_CONFIG = "mtrl_trainer/parameters/simulation.yaml"
DEFAULT_AGI_PARAM_DIR = "agilib/params"
DEFAULT_SIM_BASE_DIR = "mtrl_trainer/parameters"

DEFAULT_OUTPUT_DIR = "stratified_eval"
DEFAULT_PLOTS_DIR = "plots"

DEFAULT_NUM_EPISODES = 2000          # thesis-oriented (you can lower for iteration)
DEFAULT_MAX_STEPS = 1300             # 25s / 0.02
DEFAULT_DT = 0.02

DEFAULT_REL_POS_SOURCE = "task"
DEFAULT_REL_POS_SLICE = "0:3"
DEFAULT_REL_POS_SCALE_M = 300.0

DEFAULT_CAPTURE_RADIUS_M = 1.0
DEFAULT_NEAR_MISS_RADIUS_M = 1.5
DEFAULT_STOP_ON_CAPTURE = True

# If you KNOW your observation contains v_rel directly, set these:
# Example (common pattern): task_specific_obs[3:6] is rel velocity in normalized units
DEFAULT_VREL_SOURCE = None           # "task" or "shared" or None to use finite-diff proxy
DEFAULT_VREL_SLICE = None            # e.g. "3:6"
DEFAULT_VREL_SCALE = 30.0            # only used if DEFAULT_VREL_SOURCE is not None

DEFAULT_PRINT_EVERY = 100


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--policy_dir", default=DEFAULT_POLICY_DIR)
    ap.add_argument("--sim_config", default=DEFAULT_SIM_CONFIG)
    ap.add_argument("--agi_param_dir", default=DEFAULT_AGI_PARAM_DIR)
    ap.add_argument("--sim_base_dir", default=DEFAULT_SIM_BASE_DIR)

    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)

    ap.add_argument("--capture_radius", type=float, default=DEFAULT_CAPTURE_RADIUS_M)
    ap.add_argument("--near_miss_radius", type=float, default=DEFAULT_NEAR_MISS_RADIUS_M)

    ap.add_argument("--deterministic_evalmode", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY)
    ap.add_argument("--no_stop_on_capture", action="store_false", dest="stop_on_capture")
    ap.set_defaults(stop_on_capture=DEFAULT_STOP_ON_CAPTURE)

    # Rel pos extraction
    ap.add_argument("--rel_pos_source", choices=["task", "shared"], default=DEFAULT_REL_POS_SOURCE)
    ap.add_argument("--rel_pos_slice", default=DEFAULT_REL_POS_SLICE)
    ap.add_argument("--rel_pos_scale_m", type=float, default=DEFAULT_REL_POS_SCALE_M)

    # v_rel extraction (optional)
    ap.add_argument("--vrel_source", choices=["task", "shared"], default=DEFAULT_VREL_SOURCE)
    ap.add_argument("--vrel_slice", default=DEFAULT_VREL_SLICE)
    ap.add_argument("--vrel_scale", type=float, default=DEFAULT_VREL_SCALE)

    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    policy_dir = resolve_path(args.policy_dir)
    sim_config = resolve_path(args.sim_config)
    agi_param_dir = resolve_path(args.agi_param_dir)
    sim_base_dir = resolve_path(args.sim_base_dir)

    out_dir = resolve_path(args.output_dir)
    plots_dir = os.path.join(out_dir, DEFAULT_PLOTS_DIR)
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    print("[paths] _CURRENT_DIR  =", _CURRENT_DIR)
    print("[paths] _PROJECT_ROOT =", _PROJECT_ROOT)
    print("[paths] policy_dir    =", policy_dir)
    print("[paths] sim_config    =", sim_config)
    print("[paths] agi_param_dir =", agi_param_dir)
    print("[paths] sim_base_dir  =", sim_base_dir)
    print("[paths] output_dir    =", out_dir)

    rel_pos_sl = parse_slice(args.rel_pos_slice)

    vrel_source = args.vrel_source if args.vrel_source not in (None, "None") else None
    vrel_sl = parse_slice(args.vrel_slice) if (args.vrel_slice not in (None, "None") and vrel_source is not None) else None

    meta = {
        "policy_dir": policy_dir,
        "sim_config": sim_config,
        "agi_param_dir": agi_param_dir,
        "sim_base_dir": sim_base_dir,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "capture_radius": args.capture_radius,
        "near_miss_radius": args.near_miss_radius,
        "stop_on_capture": bool(args.stop_on_capture),
        "rel_pos_source": args.rel_pos_source,
        "rel_pos_slice": args.rel_pos_slice,
        "rel_pos_scale_m": args.rel_pos_scale_m,
        "vrel_source": vrel_source,
        "vrel_slice": args.vrel_slice,
        "vrel_scale": args.vrel_scale,
        "seed": args.seed,
        "deterministic_evalmode": bool(args.deterministic_evalmode),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[eval] Loading policy...")
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
            dt=args.dt,
            stop_on_capture=args.stop_on_capture,
            rel_pos_source=args.rel_pos_source,
            rel_pos_slice=rel_pos_sl,
            rel_pos_scale_m=args.rel_pos_scale_m,
            capture_radius_m=args.capture_radius,
            near_miss_radius_m=args.near_miss_radius,
            vrel_source=vrel_source,
            vrel_slice=vrel_sl,
            vrel_scale=args.vrel_scale,
        )
        row.episode = ep
        rows.append(row)

        if args.print_every > 0 and ((ep + 1) % args.print_every == 0):
            succ = 100.0 * float(np.mean([r.success for r in rows]))
            nm = 100.0 * float(np.mean([r.near_miss for r in rows]))
            fp = 100.0 * float(np.mean([r.first_pass_success for r in rows]))
            print(f"[{ep+1:>5}/{args.num_episodes}] success={succ:5.1f}% near={nm:5.1f}% first-pass={fp:5.1f}%")

    elapsed = time.time() - t0
    print(f"Finished {args.num_episodes} episodes in {elapsed:.1f}s ({elapsed/max(1,args.num_episodes):.2f}s/ep)")

    # Write per-episode CSV
    ep_csv = os.path.join(out_dir, "episodes.csv")
    fields = list(EpisodeRow.__annotations__.keys())
    with open(ep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"Wrote {ep_csv}")

    # Aggregate arrays
    success = np.array([r.success for r in rows], dtype=int)
    near_miss = np.array([r.near_miss for r in rows], dtype=int)
    fail = np.array([r.fail for r in rows], dtype=int)
    first_pass = np.array([r.first_pass_success for r in rows], dtype=int)

    min_dist = np.array([r.min_dist_m for r in rows], dtype=float)
    cap_t = np.array([r.capture_time_s for r in rows], dtype=float)

    init_range = np.array([r.init_range_m for r in rows], dtype=float)
    speed = np.array([r.init_vrel_norm_mps for r in rows], dtype=float)
    bearing = np.array([r.init_bearing_rad for r in rows], dtype=float)
    approach_angle = np.array([r.init_approach_angle_rad for r in rows], dtype=float)
    lateral_ratio = np.array([r.init_lateral_ratio for r in rows], dtype=float)

    succ_rate = float(np.mean(success))
    nm_rate = float(np.mean(near_miss))
    fail_rate = float(np.mean(fail))
    fp_rate = float(np.mean(first_pass))

    # Summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"episodes: {args.num_episodes}\n")
        f.write(f"success_rate (<= {args.capture_radius:.2f} m): {succ_rate*100.0:.2f}%\n")
        f.write(f"near_miss_rate (<= {args.near_miss_radius:.2f} m): {nm_rate*100.0:.2f}%\n")
        f.write(f"fail_rate: {fail_rate*100.0:.2f}%\n")
        f.write(f"first_pass_rate: {fp_rate*100.0:.2f}%\n")
        f.write(f"mean_min_dist: {float(np.mean(min_dist)):.3f} m\n")
        ct_ok = cap_t[np.isfinite(cap_t)]
        if ct_ok.size > 0:
            f.write(f"mean_capture_time_success: {float(np.mean(ct_ok)):.3f} s\n")
        f.write(f"vrel_method: {'obs' if vrel_source is not None else 'finite_difference(p_rel)'}\n")
    print(f"Wrote {summary_path}")

    # Overall stacked outcome figure
    plot_stack_overall(
        succ_rate, nm_rate, fail_rate,
        out_png=os.path.join(plots_dir, "outcome_stack.png"),
        out_pdf=os.path.join(plots_dir, "outcome_stack.pdf"),
    )

    # -----------------------------
    # Stratifications
    # -----------------------------

    # 1) Range bins (robust percentile bounds)
    r_lo, r_hi = np.nanpercentile(init_range, [1, 99])
    range_edges = np.linspace(r_lo, r_hi, 16)
    cols_range = stratify_1d(init_range, range_edges, success, near_miss, first_pass, min_dist, cap_t)
    write_strat_csv(os.path.join(out_dir, "bins_range.csv"), cols_range)
    plot_rate_with_ci(
        cols_range,
        xlabel="Initial range ||p_rel|| [m]",
        title="Success rate vs initial range (Wilson 95% CI)",
        out_png=os.path.join(plots_dir, "success_vs_range.png"),
        out_pdf=os.path.join(plots_dir, "success_vs_range.pdf"),
    )

    # 2) Speed bins (proxy)
    s_lo, s_hi = np.nanpercentile(speed, [1, 99])
    speed_edges = np.linspace(max(0.0, s_lo), s_hi, 16)
    cols_speed = stratify_1d(speed, speed_edges, success, near_miss, first_pass, min_dist, cap_t)
    write_strat_csv(os.path.join(out_dir, "bins_speed.csv"), cols_speed)
    plot_rate_with_ci(
        cols_speed,
        xlabel="Speed proxy ||v_rel|| [m/s]",
        title="Success rate vs engagement speed proxy (Wilson 95% CI)",
        out_png=os.path.join(plots_dir, "success_vs_speed.png"),
        out_pdf=os.path.join(plots_dir, "success_vs_speed.pdf"),
    )

    # 3) Encounter geometry: bearing
    bearing_edges = np.linspace(-math.pi, math.pi, 17)
    cols_bear = stratify_1d(bearing, bearing_edges, success, near_miss, first_pass, min_dist, cap_t)
    write_strat_csv(os.path.join(out_dir, "bins_bearing.csv"), cols_bear)
    plot_rate_with_ci(
        cols_bear,
        xlabel="Initial bearing atan2(y, x) [rad]",
        title="Success rate vs initial bearing (Wilson 95% CI)",
        out_png=os.path.join(plots_dir, "success_vs_bearing.png"),
        out_pdf=os.path.join(plots_dir, "success_vs_bearing.pdf"),
    )

    # 4) Encounter geometry: approach angle [0, pi]
    aa_edges = np.linspace(0.0, math.pi, 16)
    cols_aa = stratify_1d(approach_angle, aa_edges, success, near_miss, first_pass, min_dist, cap_t)
    write_strat_csv(os.path.join(out_dir, "bins_approach_angle.csv"), cols_aa)
    plot_rate_with_ci(
        cols_aa,
        xlabel="Approach angle between v_rel and -p_rel [rad]",
        title="Success rate vs approach angle (Wilson 95% CI)",
        out_png=os.path.join(plots_dir, "success_vs_approach_angle.png"),
        out_pdf=os.path.join(plots_dir, "success_vs_approach_angle.pdf"),
    )

    # 5) Encounter geometry: lateral ratio |y|/range
    lr_hi = float(np.nanpercentile(lateral_ratio, 99))
    lr_edges = np.linspace(0.0, max(0.05, lr_hi), 16)
    cols_lr = stratify_1d(lateral_ratio, lr_edges, success, near_miss, first_pass, min_dist, cap_t)
    write_strat_csv(os.path.join(out_dir, "bins_lateral_ratio.csv"), cols_lr)
    plot_rate_with_ci(
        cols_lr,
        xlabel="Lateral ratio |p_rel.y| / ||p_rel|| [-]",
        title="Success rate vs lateral ratio (Wilson 95% CI)",
        out_png=os.path.join(plots_dir, "success_vs_lateral_ratio.png"),
        out_pdf=os.path.join(plots_dir, "success_vs_lateral_ratio.pdf"),
    )

    # -----------------------------
    # Markdown report (thesis ready)
    # -----------------------------

    def md_table_from_cols(cols: Dict[str, np.ndarray], max_rows: int = 15) -> str:
        keys = ["bin_lo", "bin_hi", "count", "success_rate", "success_ci_lo", "success_ci_hi",
                "near_miss_rate", "fail_rate", "first_pass_rate", "mean_min_dist_m", "mean_capture_time_s"]
        header = "| " + " | ".join(keys) + " |\n"
        sep = "| " + " | ".join(["---"] * len(keys)) + " |\n"
        lines = [header, sep]
        n = len(cols["count"])
        for i in range(min(n, max_rows)):
            row = []
            for k in keys:
                v = cols[k][i]
                if isinstance(v, (np.floating, float)):
                    if np.isnan(v):
                        row.append("nan")
                    else:
                        row.append(f"{float(v):.3f}")
                else:
                    row.append(str(int(v)))
            lines.append("| " + " | ".join(row) + " |\n")
        return "".join(lines)

    report_md = os.path.join(out_dir, "report.md")
    with open(report_md, "w") as f:
        f.write("# Stratified interception performance\n\n")
        f.write("This section analyzes how interception performance varies with engagement conditions.\n")
        f.write("Episodes are stratified into bins by scenario parameters, and metrics are reported per bin.\n\n")

        f.write("## Overall outcomes\n\n")
        f.write(f"- Success (≤ {args.capture_radius:.2f} m): **{succ_rate*100.0:.2f}%**\n")
        f.write(f"- Near-miss ({args.capture_radius:.2f}–{args.near_miss_radius:.2f} m): **{nm_rate*100.0:.2f}%**\n")
        f.write(f"- Fail (> {args.near_miss_radius:.2f} m): **{fail_rate*100.0:.2f}%**\n")
        f.write(f"- First-pass success: **{fp_rate*100.0:.2f}%**\n\n")
        f.write(f"Velocity feature source: `{meta['vrel_method'] if 'vrel_method' in meta else ('obs' if vrel_source else 'finite_difference(p_rel)')}`\n\n")

        f.write("## Performance versus initial range\n\n")
        f.write(md_table_from_cols(cols_range))
        f.write("\n\n")

        f.write("## Performance versus target speed (proxy)\n\n")
        f.write("Speed is quantified as the norm of initial relative velocity ||v_rel||.\n")
        f.write("If v_rel is not directly available in the observation, it is estimated by finite-differencing p_rel across the first step.\n\n")
        f.write(md_table_from_cols(cols_speed))
        f.write("\n\n")

        f.write("## Performance versus encounter geometry\n\n")
        f.write("Encounter geometry is characterized by initial bearing, approach angle, and lateral ratio.\n\n")

        f.write("### Bearing\n\n")
        f.write(md_table_from_cols(cols_bear))
        f.write("\n\n")

        f.write("### Approach angle\n\n")
        f.write(md_table_from_cols(cols_aa))
        f.write("\n\n")

        f.write("### Lateral ratio\n\n")
        f.write(md_table_from_cols(cols_lr))
        f.write("\n\n")

        f.write("## Figures\n\n")
        f.write("- plots/outcome_stack.pdf\n")
        f.write("- plots/success_vs_range.pdf\n")
        f.write("- plots/success_vs_speed.pdf\n")
        f.write("- plots/success_vs_bearing.pdf\n")
        f.write("- plots/success_vs_approach_angle.pdf\n")
        f.write("- plots/success_vs_lateral_ratio.pdf\n")

    print(f"Wrote {report_md}")
    print(f"Plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
