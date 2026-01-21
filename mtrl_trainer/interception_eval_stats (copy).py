#!/usr/bin/env python3
"""
Evaluate an interception policy over many *randomised* target scenarios
(the same distribution you use in training).

We report, over NUM_EPISODES:

- capture rate (episodes where the drone gets within CAPTURE_RADIUS_M)
- time-to-capture (steps) for capture episodes
- total reward
- path length ratio (actual path / straight line start->capture-point)

IMPORTANT:
- This script runs the env in TRAINING MODE (no setEval()),
  so your C++ resetDrone(..., evalMode=false) randomisation is active.
"""

import os
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file

# Reuse scaling/unscale helpers from the visualizer
from interception_visualizer_moving import (
    OBS_POS_MIN_NP,
    OBS_POS_MAX_NP,
    unscale_position,
)

# --- Configuration for Policy Run ---
POLICY_DIR = "policies_intercept/best_intercept_850_96%"  # Path to the saved policy directory

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

# Eval config
NUM_EPISODES = 20
MAX_STEPS_PER_EPISODE = 25000
NUM_DRONES = 1  # evaluate first drone only

# Must match C++ TaskInterceptionMoving:
CAPTURE_RADIUS_M = 2       # INTERCEPTION_CAPTURE_RADIUS
REL_POS_SCALE_M = 300.0      # p_rel_world = p_rel_scaled * REL_POS_SCALE_M


# ---------------------------------------------------------------------------
# Utility: path length
# ---------------------------------------------------------------------------
def compute_path_length(traj_xyz: np.ndarray) -> float:
    """
    Euclidean path length of trajectory (N,3).
    """
    if traj_xyz.shape[0] < 2:
        return 0.0
    diffs = np.diff(traj_xyz, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


# ---------------------------------------------------------------------------
# Build env in TRAINING MODE (no setEval()!)
# ---------------------------------------------------------------------------
def build_training_env():
    track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    py_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=NUM_DRONES,
        track_layout=track_layout,
    )
    # NOTE: we deliberately DO NOT call py_env.setEval()
    # so that evalMode=false and your C++ resetDrone will randomize target.

    eval_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"[eval] Environment batch size: {eval_env.batch_size}")
    return eval_env, py_env


# ---------------------------------------------------------------------------
# Run one episode and collect stats
# ---------------------------------------------------------------------------
def run_single_episode(eval_env, policy):
    """
    Runs a single episode in training mode and returns:
      (is_capture, steps_used, total_reward, path_ratio, min_dist_m)

    - is_capture: True if min distance to target (via p_rel) <= CAPTURE_RADIUS_M
    - steps_used: total env steps (until done or max)
    - total_reward: sum over the episode
    - path_ratio: for capture episodes, path length to capture / straight start->capture
                  for non-captures, to the closest approach point
    - min_dist_m: minimum 3D distance (meters) in this episode
    """
    time_step = eval_env.reset()
    policy_state = policy.get_initial_state(eval_env.batch_size)

    drone_positions = []   # list of [x,y,z] (world)
    p_rel_scaled_list = [] # list of [x,y,z] relative (scaled)
    rewards = []
    step_count = 0

    while True:
        obs = time_step.observation
        shared_obs = obs['shared_obs'].numpy()          # (B, obs_dim)
        task_obs   = obs['task_specific_obs'].numpy()   # (B, task_dim)

        scaled_pos = shared_obs[0, :3]
        pos_world = unscale_position(scaled_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP)
        drone_positions.append(pos_world.astype(np.float32))

        p_rel_scaled = task_obs[0, :3]
        p_rel_scaled_list.append(p_rel_scaled.astype(np.float32))

        reward_val = float(time_step.reward.numpy()[0])
        rewards.append(reward_val)

        if time_step.is_last() or step_count >= MAX_STEPS_PER_EPISODE - 1:
            break

        # Policy action
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = eval_env.step(action_step.action)
        step_count += 1

    # Convert to arrays
    drone_positions = np.asarray(drone_positions, dtype=np.float32)
    p_rel_scaled_arr = np.asarray(p_rel_scaled_list, dtype=np.float32)
    total_reward = float(np.sum(rewards))
    num_steps = len(drone_positions) - 1  # transitions

    # Compute distances in meters from p_rel
    if p_rel_scaled_arr.shape[0] >= 1:
        p_rel_world = p_rel_scaled_arr * REL_POS_SCALE_M
        dists_m = np.linalg.norm(p_rel_world, axis=1)
        min_dist_m = float(dists_m.min())
        # First capture index (if any)
        hit_indices = np.where(dists_m <= CAPTURE_RADIUS_M)[0]
    else:
        dists_m = np.array([])
        min_dist_m = float('inf')
        hit_indices = np.array([], dtype=int)

    # Determine capture vs non-capture
    is_capture = hit_indices.size > 0
    if is_capture:
        cap_idx = int(hit_indices[0])
        # For capture path, truncate at capture index
        drone_path = drone_positions[:cap_idx + 1]
        final_idx = cap_idx
    else:
        # For non-capture, use path up to closest approach
        if dists_m.size > 0:
            closest_idx = int(dists_m.argmin())
        else:
            closest_idx = 0
        drone_path = drone_positions[:closest_idx + 1]
        final_idx = closest_idx

    # Path ratio: actual length / straight from start to final_idx
    if drone_path.shape[0] >= 2:
        path_len = compute_path_length(drone_path)
        straight_dist = float(np.linalg.norm(drone_positions[final_idx] - drone_positions[0]))
        path_ratio = (path_len / straight_dist) if straight_dist > 1e-6 else 0.0
    else:
        path_ratio = 0.0

    return is_capture, num_steps, total_reward, path_ratio, min_dist_m


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_policy_random_targets():
    print(f"[eval] Loading policy from: {POLICY_DIR}")
    loaded = tf.saved_model.load(POLICY_DIR)
    # In TF-Agents saved policies, the object itself is usually the policy
    # or it has a `.policy` attribute. Handle both cases:
    policy = getattr(loaded, "policy", loaded)

    eval_env, py_env = build_training_env()

    captures = 0
    steps_list = []
    total_rewards_list = []
    path_ratios_list = []
    min_dists_list = []

    for ep in range(NUM_EPISODES):
        print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")

        is_capture, num_steps, total_reward, path_ratio, min_dist_m = run_single_episode(
            eval_env, policy
        )

        steps_list.append(num_steps)
        total_rewards_list.append(total_reward)
        path_ratios_list.append(path_ratio)
        min_dists_list.append(min_dist_m)

        if is_capture:
            captures += 1
            print(f"  CAPTURE: steps={num_steps}, total_reward={total_reward:.1f}, "
                  f"min_dist={min_dist_m:.3f} m, path_ratio={path_ratio:.3f}")
        else:
            print(f"  NO CAPTURE: steps={num_steps}, total_reward={total_reward:.1f}, "
                  f"min_dist={min_dist_m:.3f} m, path_ratio={path_ratio:.3f}")

    # --- Aggregate stats ---
    if not steps_list:
        print("\nNo valid episodes evaluated.")
        return

    num_eval = len(steps_list)
    capture_rate = captures / num_eval * 100.0

    steps_arr = np.asarray(steps_list, dtype=np.float32)
    rewards_arr = np.asarray(total_rewards_list, dtype=np.float32)
    ratio_arr = np.asarray(path_ratios_list, dtype=np.float32)
    min_dists_arr = np.asarray(min_dists_list, dtype=np.float32)

    print("\n================= RANDOM-TARGET EVAL SUMMARY =================")
    print(f"Episodes evaluated: {num_eval}")
    print(f"Captures:           {captures} / {num_eval} "
          f"({capture_rate:.1f} %)")
    print(f"Steps to done:      mean={steps_arr.mean():.1f}, "
          f"std={steps_arr.std():.1f}, "
          f"min={steps_arr.min()}, "
          f"max={steps_arr.max()}")
    print(f"Total reward:       mean={rewards_arr.mean():.1f}, "
          f"std={rewards_arr.std():.1f}, "
          f"min={rewards_arr.min():.1f}, "
          f"max={rewards_arr.max():.1f}")
    print(f"Min dist (m):       mean={min_dists_arr.mean():.3f}, "
          f"std={min_dists_arr.std():.3f}, "
          f"min={min_dists_arr.min():.3f}, "
          f"max={min_dists_arr.max():.3f}")
    print(f"Path ratio:         mean={ratio_arr.mean():.3f}, "
          f"std={ratio_arr.std():.3f}, "
          f"min={ratio_arr.min():.3f}, "
          f"max={ratio_arr.max():.3f}")
    print("                     (1.0 = perfectly straight line, >1.0 = indirect)")
    print("===============================================================\n")


if __name__ == "__main__":
    evaluate_policy_random_targets()
