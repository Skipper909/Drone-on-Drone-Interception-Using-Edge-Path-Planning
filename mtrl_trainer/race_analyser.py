import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from mtrl_lib.gate_utils import load_track_from_file

# --- Python Wrapper for the C++ Environment ---
# This assumes your agisim_environment.py file is in the mtrl_lib directory
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
NUM_DRONES_PER_EVAL = 16
NUM_EVAL_STEPS = 15000  # A long horizon to ensure episodes can finish
GATE_TO_MODIFY_IDX = 2   # The index of the gate we will move (e.g., the 3rd gate)
DISPLACEMENT_AXIS = 'y'  # The axis to move the gate along ('x', 'y', or 'z')
DISPLACEMENT_STEP = 0.5 # Move the gate by 0.25 meters each time
MAX_DISPLACEMENT = 5.0   # Stop after moving the gate this far

def create_modified_track(original_track, gate_idx, displacement, axis='x'):
    """Creates a new track layout with one gate displaced along a chosen axis."""
    if gate_idx >= len(original_track):
        raise ValueError("gate_idx is out of bounds for the track layout.")

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("Displacement axis must be 'x', 'y', or 'z'.")

    # Create a deep copy to avoid modifying the original track
    modified_track = copy.deepcopy(original_track)

    # Apply the displacement to each of the 4 corners of the selected gate
    for corner in modified_track[gate_idx]:
        corner[axis_map[axis]] += displacement

    return modified_track

def run_evaluation(policy, env_params, track_layout, num_drones, num_steps):
    """Runs a single evaluation and returns crash rate, avg reward, and avg episode length."""
    if BatchedAgiSimEnv is None: return 100.0, 0.0, 0.0 # Return 100% crash rate if env fails

    try:
        py_eval_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=num_drones,
            track_layout=track_layout
        )
        eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
    except Exception as e:
        print(f"  Error initializing environment: {e}")
        return 100.0, 0.0, 0.0

    # --- MODIFIED: Add tracking for episode steps ---
    completed_episode_outcomes = []
    completed_episode_rewards = []
    completed_episode_lengths = [] # New list to store duration of each episode
    current_episode_rewards = np.zeros(eval_env.batch_size, dtype=np.float32)
    current_episode_steps = np.zeros(eval_env.batch_size, dtype=np.int32) # New array to track steps

    eval_env.setEval()
    time_step = eval_env.reset()
    policy_state = policy.get_initial_state(eval_env.batch_size)

    for _ in range(num_steps):
        action_step = policy.action(time_step, policy_state)
        time_step = eval_env.step(action_step.action)

        success_batch = eval_env.pyenv.get_last_success_flags()
        is_last_batch = time_step.is_last().numpy()
        rewards_batch = time_step.reward.numpy()

        for i in range(eval_env.batch_size):
            current_episode_rewards[i] += rewards_batch[i]
            current_episode_steps[i] += 1 # Increment step count for this drone
            if is_last_batch[i]:
                # Episode ended, record all stats
                completed_episode_outcomes.append('success' if success_batch[i] else 'crash')
                completed_episode_rewards.append(current_episode_rewards[i])
                completed_episode_lengths.append(current_episode_steps[i])
                # Reset counters for the new episode
                current_episode_rewards[i] = 0
                current_episode_steps[i] = 0

        policy_state = action_step.state

    eval_env.close()

    num_episodes_completed = len(completed_episode_outcomes)
    if num_episodes_completed == 0:
        return 0.0, 0.0, 0.0

    # --- MODIFIED SECTION ---
    # Filter the lengths to include only those from successful episodes
    successful_episode_lengths = [
        length for i, length in enumerate(completed_episode_lengths)
        if completed_episode_outcomes[i] == 'success'
    ]

    num_crashed_episodes = completed_episode_outcomes.count('crash')
    crash_rate = (num_crashed_episodes / num_episodes_completed) * 100
    avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0

    # Calculate the average length using only the successful episodes
    avg_length = np.mean(successful_episode_lengths) if successful_episode_lengths else 0
    # --- END OF MODIFIED SECTION ---
    # Return the new metric
    return crash_rate, avg_reward, avg_length

def plot_crash_rate_vs_displacement(displacements, crash_rates):
    """Generates and displays the final plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(displacements, crash_rates, marker='o', linestyle='-', color='r')
    plt.xlabel(f'Gate Displacement (meters)')
    plt.ylabel('Crash Rate (%)')
    plt.grid(True)
    plt.ylim(0, 105)
    plt.show()

def plot_analysis_results(displacements, crash_rates, avg_rewards, avg_lengths):
    """Generates and displays final plots for crash rate, reward, and episode length."""

    # Create a figure with 3 subplots, stacked vertically, sharing the same x-axis
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # --- Plot 1: Crash Rate ---
    ax1.plot(displacements, crash_rates, marker='o', linestyle='-', color='r')
    ax1.set_ylabel('Crash Rate (%)')
    ax1.grid(True)
    ax1.set_ylim(0, 105)


    # --- Plot 3: Average Episode Length ---
    ax3.plot(displacements, avg_lengths, marker='o', linestyle='-', color='g')
    ax3.set_ylabel('Average Time Steps (2 Laps)')
    ax3.set_xlabel('Gate Displacement (m)')
    ax3.grid(True)

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

    displacements = []
    crash_rates = []
    avg_rewards = []
    avg_lengths = [] # New list to store the average lengths

    # Loop over different displacement amounts
    for displacement in np.arange(0, MAX_DISPLACEMENT + DISPLACEMENT_STEP, DISPLACEMENT_STEP):
        print(f"\n--- Running evaluation for displacement: {displacement:.2f}m ---")
        displacements.append(displacement)

        # 1. Create the modified track for this run
        modified_track = create_modified_track(final_track_layout, GATE_TO_MODIFY_IDX, displacement, DISPLACEMENT_AXIS)

        # 2. Run the evaluation and get all three statistics
        crash_rate, avg_reward, avg_length = run_evaluation(
            loaded_policy, env_params, modified_track, NUM_DRONES_PER_EVAL, NUM_EVAL_STEPS
        )
        crash_rates.append(crash_rate)
        avg_rewards.append(avg_reward)
        avg_lengths.append(avg_length) # Store the new metric

        print(f"  Result: Crash Rate = {crash_rate:.2f}%, Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_length:.1f}")

        # 3. Stop if the crash rate reaches 100%
        if crash_rate >= 100.0:
            print("\nCrash rate reached 100%. Stopping analysis.")
            break

    print("\n--- Analysis Complete. Generating plots... ---")
    plot_analysis_results(displacements, crash_rates, avg_rewards, avg_lengths)

if __name__ == "__main__":
    main()