#!/usr/bin/env python3
"""
Run multiple randomised interception episodes (training-style eval),
collect drone & target trajectories, and generate ONE GIF that overlays
the paths from several episodes (up to MAX_EPISODES_IN_GIF).

- Uses the same env config as your training.
- DOES NOT call setEval(), so C++ resetDrone(..., evalMode=false)
  randomisation (target position + velocity) is active.

Key behavior fix vs your previous version:
- The episode now TERMINATES immediately when CAPTURE_RADIUS_M is reached
  (Python-side), regardless of whether the C++ env terminates on capture.
- We also record capture_step and print whether env_last was true at capture.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from matplotlib.animation import FuncAnimation, PillowWriter

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file

# --- Configuration for Policy / Env paths (match your other scripts) ---

POLICY_DIR = "policies_intercept/best_intercept_850_96%"  # Path to the saved policy directory

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "mtrl_trainer", "parameters", "simulation.yaml")
AGI_PARAM_DIR   = os.path.join(_PROJECT_ROOT, "agilib", "params")
SIM_BASE_DIR    = os.path.join(_PROJECT_ROOT, "mtrl_trainer", "parameters")

# --- Eval config ---

NUM_EPISODES           = 10           # how many random scenarios to run
MAX_EPISODES_IN_GIF    = 5            # how many to overlay in the GIF
MAX_STEPS_PER_EPISODE  = 25000        # safety cap
CAPTURE_RADIUS_M       = 2.0          # must match INTERCEPTION_CAPTURE_RADIUS (or whatever you're using)

# Position scaling for shared_obs (same as interception_visualizer_moving)
OBS_POS_MIN_NP = np.array([-200.0, -200.0, -200.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([ 200.0,  200.0,  200.0], dtype=np.float32)

# p_rel scaling must match POS_SCALE in TaskInterceptionMoving::getTaskSpecificObservation:
#   p_rel_scaled = p_rel_world / POS_SCALE
#   => p_rel_world = p_rel_scaled * POS_SCALE
REL_POS_SCALE_M = 300.0   # <-- set this to your C++ POS_SCALE (200.0 or 300.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unscale_position(scaled_pos_np: np.ndarray,
                     min_bounds_np: np.ndarray,
                     max_bounds_np: np.ndarray) -> np.ndarray:
    """
    Unscale a 3D position vector from [-1,1] back to world coordinates.
    Same as in interception_visualizer_moving.py.
    """
    range_bounds = max_bounds_np - min_bounds_np
    range_bounds = np.where(np.isclose(range_bounds, 0.0), 1.0, range_bounds)
    unscaled_pos = ((scaled_pos_np + 1.0) * range_bounds / 2.0) + min_bounds_np
    return unscaled_pos


def compute_path_length(traj_xyz: np.ndarray) -> float:
    """
    Euclidean path length of trajectory (N,3).
    """
    if traj_xyz.shape[0] < 2:
        return 0.0
    diffs = np.diff(traj_xyz, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def build_training_env():
    """
    Build the BatchedAgiSimEnv in TRAINING MODE (no setEval()).
    """
    track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    py_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=1,
        track_layout=track_layout,
    )

    # IMPORTANT: do NOT call setEval() here, we want evalMode=false
    # so TaskInterceptionMoving::resetDrone uses the randomised logic.

    eval_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"[multi-gif] Environment batch size: {eval_env.batch_size}")
    return eval_env, py_env


def run_single_episode_collect(eval_env, policy, episode_idx: int):
    """
    Run one episode in training mode and collect:
      - drone_traj: (T,3) drone positions in world
      - target_traj: (T,3) target positions in world (from scaled p_rel)
      - capture_flag: True if dist <= CAPTURE_RADIUS_M at any step
      - capture_step: step index where capture first happened (None if not captured)
      - min_dist: minimum 3D distance [m]
      - steps_used: number of env steps executed (actions applied)
      - path_ratio: actual path length / straight line from start->final
    """
    time_step = eval_env.reset()
    policy_state = policy.get_initial_state(eval_env.batch_size)

    drone_traj = []
    target_traj = []
    dists = []

    step_count = 0
    capture_flag = False
    capture_step = None
    env_last_at_capture = None

    while True:
        shared_obs = time_step.observation["shared_obs"].numpy()         # (B, dim)
        task_obs   = time_step.observation["task_specific_obs"].numpy()  # (B, dim)

        # Use drone 0 (batch size is 1)
        scaled_pos = shared_obs[0, :3]
        pos_world  = unscale_position(scaled_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP)

        # p_rel (scaled) is first 3 dims in task_specific_obs
        p_rel_scaled = task_obs[0, 0:3]
        p_rel_world  = p_rel_scaled * REL_POS_SCALE_M
        target_world = pos_world + p_rel_world

        # Store current step positions
        drone_traj.append(pos_world.astype(np.float32))
        target_traj.append(target_world.astype(np.float32))

        # Distance (3D)
        dist = float(np.linalg.norm(target_world - pos_world))
        dists.append(dist)

        # Capture logic (Python-side termination)
        if (not capture_flag) and (dist <= CAPTURE_RADIUS_M):
            capture_flag = True
            capture_step = step_count
            env_last_at_capture = bool(time_step.is_last())
            print(f"[ep {episode_idx+1}] CAPTURE at step={capture_step} "
                  f"(dist={dist:.3f} m, env_last={env_last_at_capture})")
            # Terminate immediately on first capture (this is the fix)
            break

        # Normal termination
        if time_step.is_last():
            break
        if step_count >= MAX_STEPS_PER_EPISODE - 1:
            break

        # Policy action
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = eval_env.step(action_step.action)
        step_count += 1

    drone_np = np.asarray(drone_traj, dtype=np.float32)
    target_np = np.asarray(target_traj, dtype=np.float32)
    dists_np  = np.asarray(dists, dtype=np.float32)

    min_dist = float(dists_np.min()) if dists_np.size > 0 else float("inf")

    # steps_used is the number of actions actually applied (step_count),
    # while len(drone_traj) is the number of state samples collected (step_count+1 typically).
    steps_used = step_count

    # Path ratio: actual / straight from start to final drone point
    if drone_np.shape[0] >= 2:
        path_len = compute_path_length(drone_np)
        straight_dist = float(np.linalg.norm(drone_np[-1] - drone_np[0]))
        path_ratio = (path_len / straight_dist) if straight_dist > 1e-6 else 0.0
    else:
        path_ratio = 0.0

    print(f"\n=== Episode {episode_idx+1} ===")
    print(f"  steps_used: {steps_used}")
    print(f"  min_dist:   {min_dist:.3f} m")
    print(f"  path_ratio: {path_ratio:.3f}")
    print(f"  capture:    {capture_flag} (radius={CAPTURE_RADIUS_M} m)")
    if capture_flag:
        print(f"  capture_step: {capture_step}, env_last_at_capture: {env_last_at_capture}")

    return dict(
        drone=drone_np,
        target=target_np,
        capture=capture_flag,
        capture_step=capture_step,
        env_last_at_capture=env_last_at_capture,
        min_dist=min_dist,
        steps=steps_used,
        path_ratio=path_ratio,
        episode_idx=episode_idx,
    )


def make_multi_episode_gif(episodes,
                           output_path="interception_eval_multi.gif",
                           fps=20,
                           frame_stride=2):
    """
    Create a single 3D GIF showing the overlaid paths from multiple episodes.

    episodes: list of dicts as returned by run_single_episode_collect
              (we will use up to MAX_EPISODES_IN_GIF of them).
    """
    if not episodes:
        print("[multi-gif] No episodes provided, skipping GIF.")
        return

    # Limit to the first MAX_EPISODES_IN_GIF
    episodes = episodes[:MAX_EPISODES_IN_GIF]
    num_plot = len(episodes)
    print(f"[multi-gif] Making GIF for {num_plot} episodes...")

    # Subsample trajectories and compute global axis limits
    subsampled_eps = []
    all_points = []

    for ep in episodes:
        drone = ep["drone"]
        target = ep["target"]

        min_len = min(len(drone), len(target))
        if min_len < 2:
            continue

        indices = np.arange(0, min_len, frame_stride, dtype=int)
        drone_sub = drone[indices]
        target_sub = target[indices]

        subsampled_eps.append({
            "drone": drone_sub,
            "target": target_sub,
            "capture": ep["capture"],
            "capture_step": ep.get("capture_step", None),
            "min_dist": ep["min_dist"],
            "path_ratio": ep["path_ratio"],
            "episode_idx": ep["episode_idx"],
        })

        all_points.append(drone_sub)
        all_points.append(target_sub)

    if not subsampled_eps:
        print("[multi-gif] Not enough points after subsampling, skipping GIF.")
        return

    all_pts = np.vstack(all_points)
    margin = 10.0
    x_min, y_min, z_min = all_pts.min(axis=0) - margin
    x_max, y_max, z_max = all_pts.max(axis=0) + margin

    # Determine max number of frames across episodes
    max_frames = max(len(ep["drone"]) for ep in subsampled_eps)

    # ---- Setup 3D plot ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Interception Trajectories (Multiple Episodes)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Use a color map to give each episode its own color
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(subsampled_eps))]

    drone_lines = []
    target_lines = []
    drone_pts = []
    target_pts = []

    for i, ep in enumerate(subsampled_eps):
        color = colors[i]
        (dr_line,) = ax.plot([], [], [], "-", linewidth=1.5, color=color,
                             label=f"Drone Ep {ep['episode_idx']+1}")
        (tg_line,) = ax.plot([], [], [], "--", linewidth=1.0, color=color, alpha=0.7)
        (dr_pt,) = ax.plot([], [], [], "o", color=color, markersize=5)
        (tg_pt,) = ax.plot([], [], [], "s", color=color, markersize=4, alpha=0.7)

        drone_lines.append(dr_line)
        target_lines.append(tg_line)
        drone_pts.append(dr_pt)
        target_pts.append(tg_pt)

    ax.legend(loc="upper left")

    def init():
        for dr_line, tg_line, dr_pt, tg_pt in zip(drone_lines, target_lines, drone_pts, target_pts):
            dr_line.set_data([], [])
            dr_line.set_3d_properties([])
            tg_line.set_data([], [])
            tg_line.set_3d_properties([])
            dr_pt.set_data([], [])
            dr_pt.set_3d_properties([])
            tg_pt.set_data([], [])
            tg_pt.set_3d_properties([])
        return drone_lines + target_lines + drone_pts + target_pts

    def update(frame_idx):
        for i, ep in enumerate(subsampled_eps):
            drone = ep["drone"]
            target = ep["target"]

            # Clamp frame index to episode length - 1 (short episodes "freeze")
            idx = min(frame_idx, len(drone) - 1)

            xs_d, ys_d, zs_d = drone[: idx + 1].T
            xs_t, ys_t, zs_t = target[: idx + 1].T

            drone_lines[i].set_data(xs_d, ys_d)
            drone_lines[i].set_3d_properties(zs_d)

            target_lines[i].set_data(xs_t, ys_t)
            target_lines[i].set_3d_properties(zs_t)

            drone_pts[i].set_data([xs_d[-1]], [ys_d[-1]])
            drone_pts[i].set_3d_properties([zs_d[-1]])

            target_pts[i].set_data([xs_t[-1]], [ys_t[-1]])
            target_pts[i].set_3d_properties([zs_t[-1]])

        return drone_lines + target_lines + drone_pts + target_pts

    ani = FuncAnimation(
        fig,
        update,
        frames=np.arange(max_frames),
        init_func=init,
        blit=False,
        interval=1000.0 / fps,
    )

    print(f"[multi-gif] Saving GIF to '{output_path}' (fps={fps})...")
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print("[multi-gif] Done:", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[multi-gif] Loading policy from: {POLICY_DIR}")
    loaded = tf.saved_model.load(POLICY_DIR)
    policy = getattr(loaded, "policy", loaded)

    eval_env, _ = build_training_env()

    episodes = []
    for ep in range(NUM_EPISODES):
        ep_stats = run_single_episode_collect(eval_env, policy, ep)
        episodes.append(ep_stats)

    # Create one GIF with multiple episodes overlaid
    gif_path = os.path.join(_CURRENT_DIR, "interception_eval_multi.gif")
    make_multi_episode_gif(episodes, output_path=gif_path, fps=20, frame_stride=2)

    # Print a brief summary
    num_cap = sum(1 for ep in episodes if ep["capture"])
    print("\n==== Summary over episodes ====")
    print(f"Episodes: {len(episodes)}")
    print(f"Captures: {num_cap} / {len(episodes)} "
          f"({100.0 * num_cap / len(episodes):.1f}%)")
    for ep in episodes[:MAX_EPISODES_IN_GIF]:
        cap = "CAP" if ep["capture"] else "NO"
        cap_step = ep["capture_step"]
        cap_step_str = f"{cap_step}" if cap_step is not None else "-"
        print(f"  Ep {ep['episode_idx']+1:02d}: "
              f"{cap} | "
              f"min={ep['min_dist']:.2f} m | "
              f"ratio={ep['path_ratio']:.3f} | "
              f"cap_step={cap_step_str}")


if __name__ == "__main__":
    main()
