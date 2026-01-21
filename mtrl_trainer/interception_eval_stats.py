#!/usr/bin/env python3
"""
Random-target evaluation for TaskInterceptionMoving.

Fixes:
- Robust policy loading: GreedyPolicy only if the loaded object supports it.
- Works with TF-Agents PolicySaver SavedModel objects (_UserObject).
- Robust action call: supports policy.action(ts, state) OR policy.action(ts).
- Reconstruct target world position from p_rel in task_specific_obs:
    target_world = drone_world + p_rel_world
  so target motion is represented correctly.

Reports:
- CAPTURE if min_dist <= CAPTURE_RADIUS_M (independent of env termination)
- steps_to_cap, min_dist, total_reward, path_ratio (to capture point if captured)
"""

import os
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

try:
    from tf_agents.policies import greedy_policy
except Exception:
    greedy_policy = None

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file

from interception_visualizer_moving import (
    OBS_POS_MIN_NP,
    OBS_POS_MAX_NP,
    unscale_position,
)

# -----------------------------------------------------------------------------
# --- Configuration for Policy Run ---
POLICY_DIR = "policies_intercept/best_intercept_10msTarget"  # Path to the saved policy directory

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')
# -----------------------------------------------------------------------------

NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 25000
NUM_DRONES = 1

# Must match C++ POS_SCALE used for p_rel scaling in TaskInterceptionMoving
REL_POS_SCALE_M = 300.0

# Capture threshold for eval counting (you want 1 m)
CAPTURE_RADIUS_M = 1

# If True: call setEval() => deterministic target reset (evalMode=true)
# If False: do NOT call setEval() => training randomization active (evalMode=false)
DETERMINISTIC_EVALMODE = False


def _read_simple_yaml_value(path: str, key: str, fallback: float) -> float:
    """Best-effort parse 'KEY: value' without PyYAML."""
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if not line.startswith(key):
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                val_str = parts[1].strip()
                return float(val_str)
    except Exception:
        return fallback
    return fallback


def load_interception_reward_params():
    reward_yaml_path = os.path.join(SIM_BASE_DIR, "reward_params.yaml")
    return {
        "capture_radius": _read_simple_yaml_value(reward_yaml_path, "INTERCEPTION_CAPTURE_RADIUS", CAPTURE_RADIUS_M),
        "success_reward": _read_simple_yaml_value(reward_yaml_path, "INTERCEPTION_SUCCESS_REWARD", 500.0),
        "max_time_s": _read_simple_yaml_value(reward_yaml_path, "INTERCEPTION_MAX_EPISODE_TIME_S", 30.0),
    }


def compute_path_length(traj_xyz: np.ndarray) -> float:
    if traj_xyz.shape[0] < 2:
        return 0.0
    diffs = np.diff(traj_xyz, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def build_env():
    track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    py_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=NUM_DRONES,
        track_layout=track_layout,
    )

    if DETERMINISTIC_EVALMODE and hasattr(py_env, "setEval"):
        print("[eval] setEval() enabled => deterministic target reset (evalMode=true)")
        py_env.setEval()
    else:
        print("[eval] setEval() NOT called => training randomization active (evalMode=false)")

    eval_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"[eval] Environment batch size: {eval_env.batch_size}")
    return eval_env


def load_policy(policy_dir: str):
    if not tf.io.gfile.exists(policy_dir):
        raise FileNotFoundError(f"[eval] POLICY_DIR does not exist: {policy_dir}")

    loaded = tf.saved_model.load(policy_dir)
    base = getattr(loaded, "policy", loaded)

    # Only wrap with GreedyPolicy if it actually supports TF-Agents Policy API
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
    # Fallback: empty state
    return ()


def policy_action(policy, time_step, policy_state):
    """
    Support both:
      policy.action(time_step, policy_state)
      policy.action(time_step)
    """
    try:
        return policy.action(time_step, policy_state)
    except TypeError:
        return policy.action(time_step)


def run_single_episode(eval_env, policy, capture_radius_m: float):
    time_step = eval_env.reset()
    policy_state = get_initial_state(policy, eval_env.batch_size)

    drone_positions = []
    target_positions = []
    rewards = []

    min_dist_m = float("inf")
    capture_step = None

    step_count = 0
    while True:
        obs = time_step.observation
        shared_obs = obs["shared_obs"].numpy()          # (B, D)
        task_obs = obs["task_specific_obs"].numpy()     # (B, K)

        # Drone world position from shared_obs (scaled -> unscaled)
        scaled_pos = shared_obs[0, :3]
        drone_p_world = unscale_position(scaled_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP).astype(np.float32)

        # Target relative position from task_specific_obs (scaled -> meters)
        # Assumes first 3 elements are p_rel_scaled.
        p_rel_scaled = task_obs[0, 0:3].astype(np.float32)
        p_rel_world = p_rel_scaled * REL_POS_SCALE_M

        target_p_world = drone_p_world + p_rel_world

        dist_m = float(np.linalg.norm(target_p_world - drone_p_world))
        min_dist_m = min(min_dist_m, dist_m)

        drone_positions.append(drone_p_world)
        target_positions.append(target_p_world)
        rewards.append(float(time_step.reward.numpy()[0]))

        if capture_step is None and dist_m <= capture_radius_m:
            capture_step = step_count

        if time_step.is_last() or step_count >= MAX_STEPS_PER_EPISODE - 1:
            break

        action_step = policy_action(policy, time_step, policy_state)

        # Update state if present
        if hasattr(action_step, "state"):
            policy_state = action_step.state

        time_step = eval_env.step(action_step.action)
        step_count += 1

    drone_positions = np.asarray(drone_positions, dtype=np.float32)
    total_reward = float(np.sum(rewards))
    steps_used = int(drone_positions.shape[0] - 1)

    end_idx = capture_step if capture_step is not None else (drone_positions.shape[0] - 1)

    if end_idx >= 1:
        path_len = compute_path_length(drone_positions[: end_idx + 1])
        straight_dist = float(np.linalg.norm(drone_positions[end_idx] - drone_positions[0]))
        path_ratio = (path_len / straight_dist) if straight_dist > 1e-6 else 0.0
    else:
        path_ratio = 0.0

    is_capture = capture_step is not None
    return is_capture, steps_used, total_reward, min_dist_m, path_ratio, capture_step


def main():
    rp = load_interception_reward_params()
    print(
        "[eval] reward_params.yaml: "
        f"INTERCEPTION_CAPTURE_RADIUS={rp['capture_radius']}, "
        f"INTERCEPTION_SUCCESS_REWARD={rp['success_reward']}, "
        f"INTERCEPTION_MAX_EPISODE_TIME_S={rp['max_time_s']}"
    )
    print(f"[eval] Using CAPTURE_RADIUS_M={CAPTURE_RADIUS_M} for capture counting.")
    print(f"[eval] Loading policy from: {POLICY_DIR}")

    policy = load_policy(POLICY_DIR)
    eval_env = build_env()

    captures = 0
    steps_done = []
    steps_to_cap = []
    total_rewards = []
    min_dists = []
    path_ratios = []

    for ep in range(NUM_EPISODES):
        is_cap, steps_used, tot_rew, min_dist, path_ratio, cap_step = run_single_episode(
            eval_env, policy, CAPTURE_RADIUS_M
        )

        steps_done.append(steps_used)
        total_rewards.append(tot_rew)
        min_dists.append(min_dist)
        path_ratios.append(path_ratio)

        print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")
        if is_cap:
            captures += 1
            steps_to_cap.append(int(cap_step))
            print(f"  CAPTURE: steps_done={steps_used}, steps_to_cap={cap_step}, total_reward={tot_rew:.1f}, "
                  f"min_dist={min_dist:.3f} m, path_ratio={path_ratio:.3f}")
        else:
            print(f"  NO CAPTURE: steps_done={steps_used}, total_reward={tot_rew:.1f}, "
                  f"min_dist={min_dist:.3f} m, path_ratio={path_ratio:.3f}")

    steps_done = np.asarray(steps_done, dtype=np.float32)
    total_rewards = np.asarray(total_rewards, dtype=np.float32)
    min_dists = np.asarray(min_dists, dtype=np.float32)
    path_ratios = np.asarray(path_ratios, dtype=np.float32)
    steps_to_cap_arr = np.asarray(steps_to_cap, dtype=np.float32) if steps_to_cap else None

    print("\n================= RANDOM-TARGET EVAL SUMMARY =================")
    print(f"Episodes evaluated: {NUM_EPISODES}")
    print(f"Captures:           {captures} / {NUM_EPISODES} ({captures/NUM_EPISODES*100.0:.1f} %)")
    print(f"Steps to done:      mean={steps_done.mean():.1f}, std={steps_done.std():.1f}, "
          f"min={steps_done.min()}, max={steps_done.max()}")
    if steps_to_cap_arr is not None:
        print(f"Steps to capture:   mean={steps_to_cap_arr.mean():.1f}, std={steps_to_cap_arr.std():.1f}, "
              f"min={steps_to_cap_arr.min():.0f}, max={steps_to_cap_arr.max():.0f}")
    else:
        print("Steps to capture:   n/a (no captures)")
    print(f"Total reward:       mean={total_rewards.mean():.1f}, std={total_rewards.std():.1f}, "
          f"min={total_rewards.min():.1f}, max={total_rewards.max():.1f}")
    print(f"Min dist (m):       mean={min_dists.mean():.3f}, std={min_dists.std():.3f}, "
          f"min={min_dists.min():.3f}, max={min_dists.max():.3f}")
    print(f"Path ratio:         mean={path_ratios.mean():.3f}, std={path_ratios.std():.3f}, "
          f"min={path_ratios.min():.3f}, max={path_ratios.max():.3f}")
    print("                     (1.0 = perfectly straight line, >1.0 = indirect)")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
