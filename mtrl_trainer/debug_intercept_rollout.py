import os
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file

# ------------------------------------------------
# Paths
# ------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

SIM_CONFIG_PATH = os.path.join(
    _PROJECT_ROOT, "mtrl_trainer", "parameters", "simulation.yaml"
)

AGI_PARAM_DIR = os.path.join(
    _PROJECT_ROOT, "agilib", "params"
)

SIM_BASE_DIR = os.path.join(
    _PROJECT_ROOT, "mtrl_trainer", "parameters"
)

# Pick a saved interception policy
POLICY_DIR = os.path.join(_CURRENT_DIR, "policies_intercept", "best_intercept2")
if not os.path.exists(os.path.join(POLICY_DIR, "saved_model.pb")):
    raise FileNotFoundError(f"saved_model.pb not found in {POLICY_DIR}")

TRACK_PATH = os.path.join(SIM_BASE_DIR, "track.json")


def main():
    # Load track (same helper as training script)
    track_layout = load_track_from_file(TRACK_PATH)

    # 1 drone for debugging
    py_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=1,
        track_layout=track_layout,
    )

    if hasattr(py_env, "setEval"):
        py_env.setEval()

    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # Load saved policy
    policy = tf.compat.v2.saved_model.load(POLICY_DIR)

    time_step = tf_env.reset()

    # Print initial shapes
    obs0 = time_step.observation
    print("Initial observation shapes:")
    for k, v in obs0.items():
        print(f"  {k}: {v.shape}")

    print("\nRolling out one episode...\n")

    max_steps = 500
    step_idx = 0

    while not time_step.is_last() and step_idx < max_steps:
        # Convert obs to numpy
        obs_np = {k: v.numpy() for k, v in time_step.observation.items()}

        # Remove batch dim (num_drones=1 -> shape (1, N))
        shared = np.squeeze(obs_np["shared_obs"], axis=0)         # (19,)
        task   = np.squeeze(obs_np["task_specific_obs"], axis=0)  # (24,)

        # shared[0:3] = drone position (x,y,z)
        drone_pos = shared[0:3]

        # task[0:3] = p_rel (target - drone), task[6] = dist (already scaled by C++)
        p_rel = task[0:3]
        dist  = task[6]

        reward = float(time_step.reward.numpy()[0])

        print(
            f"t={step_idx:3d}  "
            f"p=({drone_pos[0]:7.2f}, {drone_pos[1]:7.2f}, {drone_pos[2]:7.2f})  "
            f"p_rel=({p_rel[0]:7.2f}, {p_rel[1]:7.2f}, {p_rel[2]:7.2f})  "
            f"dist={dist:7.2f}  "
            f"r={reward:7.3f}"
        )

        # Step policy + env
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)

        step_idx += 1

    print("\nEpisode done.")


if __name__ == "__main__":
    main()
