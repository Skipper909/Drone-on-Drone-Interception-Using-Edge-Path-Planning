import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from mtrl_lib.gate_utils import load_track_from_file

# Import your custom environment wrapper
try:
    from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv.")
    BatchedAgiSimEnv = None

# --- Configuration ---
POLICY_DIR = "policies/ppo_policy_mtrl"



_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters')



# --- Analysis Parameters ---
NUM_DRONES_PER_EVAL = 2
NUM_EVAL_STEPS = 6000  # A long horizon to ensure episodes can finish
NUM_INTENSITY_LEVELS = 8 # e.g., 0% to 100% in 10% steps

# Define the MAXIMUM noise values for 100% intensity
# These values represent the standard deviation of the noise.
MAX_VIO_POS_DRIFT_STD = 0.03   # meters of drift added per step
MAX_VIO_ATT_DRIFT_STD = 0.8   # degrees of drift added per step
MAX_GATE_RESET_POS_STD = 0.0   # meters of uncertainty after a global position update
MAX_GATE_RESET_ATT_STD = 0.0   # degrees of uncertainty after a global position update

def run_evaluation(policy, env_params, pos_drift_std, att_drift_std_deg, gate_pos_std, gate_att_std_deg, track_layout):
    """Runs a single evaluation for a given track layout and returns the crash rate."""
    if BatchedAgiSimEnv is None: return 100.0, 0.0 # Return 100% crash rate if env fails




    try:
        py_eval_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=NUM_DRONES_PER_EVAL,
            track_layout=track_layout
        )
        eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
    except Exception as e:
        print(f"  Error initializing environment: {e}")
        return 100.0, 0.0

    eval_env.setNoiseIntensity(
        pos_drift_std, att_drift_std_deg, gate_pos_std, gate_att_std_deg
    )

    # Per-episode tracking logic
    completed_episode_outcomes = []
    completed_episode_rewards = []
    current_episode_rewards = np.zeros(eval_env.batch_size, dtype=np.float32)


    eval_env.setEval()
    time_step = eval_env.reset()
    policy_state = policy.get_initial_state(eval_env.batch_size)

    for _ in range(NUM_EVAL_STEPS):
        action_step = policy.action(time_step, policy_state)
        time_step = eval_env.step(action_step.action)


        success_batch = eval_env.pyenv.get_last_success_flags()
        is_last_batch = time_step.is_last().numpy()
        rewards_batch = time_step.reward.numpy()

        for i in range(eval_env.batch_size):
            current_episode_rewards[i] += rewards_batch[i]
            if is_last_batch[i]:
                completed_episode_outcomes.append('success' if success_batch[i] else 'crash')
                completed_episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0

        policy_state = action_step.state

    eval_env.close()

    num_episodes_completed = len(completed_episode_outcomes)
    if num_episodes_completed == 0:
        return 0.0, 0.0 # No episodes finished, assume 0% crash rate for this run

    num_crashed_episodes = completed_episode_outcomes.count('crash')
    crash_rate = (num_crashed_episodes / num_episodes_completed) * 100
    avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0

    return crash_rate, avg_reward

def plot_results(intensity_percentages, crash_rates, avg_rewards):
    """Generates and displays the final plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Policy Robustness vs. VIO Drift & Noise Intensity', fontsize=16)

    # Plot 1: Crash Rate
    ax1.plot(intensity_percentages, crash_rates, marker='o', linestyle='-', color='r')
    ax1.set_ylabel('Crash Rate (%)')
    ax1.grid(True)
    ax1.set_ylim(0, 105)

    # Plot 2: Average Reward
    ax2.plot(intensity_percentages, avg_rewards, marker='o', linestyle='-', color='b')
    ax2.set_ylabel('Average Reward')
    ax2.set_xlabel('Noise Intensity (%)')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    if not os.path.isdir(POLICY_DIR) or BatchedAgiSimEnv is None:
        print(f"ERROR: Policy directory '{POLICY_DIR}' not found or environment not available.")
        return

    print(f"--- Loading Policy from: {POLICY_DIR} ---")
    try:
        loaded_policy = tf.saved_model.load(POLICY_DIR)
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    env_params = {
        'sim_config_path': SIM_CONFIG_PATH,
        'agi_param_dir': AGI_PARAM_DIR,
        'sim_base_dir': SIM_BASE_DIR,
    }

    final_track_layout = load_track_from_file(SIM_BASE_DIR + "/track.json")

    # Main Analysis Loop
    intensity_levels = np.linspace(0, 1.0, NUM_INTENSITY_LEVELS)
    crash_rates = []
    avg_rewards = []

    for intensity in intensity_levels:
        print(f"\n--- Running evaluation for Noise Intensity: {intensity*100:.0f}% ---")

        # 1. Calculate the noise parameters for this intensity level
        pos_drift_std = intensity * MAX_VIO_POS_DRIFT_STD
        att_drift_std_deg = intensity * MAX_VIO_ATT_DRIFT_STD
        gate_pos_std = intensity * MAX_GATE_RESET_POS_STD
        gate_att_std_deg = intensity * MAX_GATE_RESET_ATT_STD

        # 2. Set the noise level in the C++ environment via the pybind11 function


        # 3. Run the evaluation and get statistics
        crash_rate, avg_reward = run_evaluation(loaded_policy, env_params, pos_drift_std, att_drift_std_deg, gate_pos_std, gate_att_std_deg, final_track_layout)

        crash_rates.append(crash_rate)
        avg_rewards.append(avg_reward)

        print(f"  Result: Crash Rate = {crash_rate:.2f}%, Average Reward = {avg_reward:.2f}")

    # 4. Plot the final results
    print("\n--- Analysis Complete. Generating plots... ---")
    plot_results(intensity_levels * 100, crash_rates, avg_rewards)


if __name__ == "__main__":
    main()
