import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import json

# --- TF-Agents and TensorFlow Imports ---
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

try:
    # Ensure this path is correct or the module is in PYTHONPATH
    from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv.")
    print("Please ensure this file is in the Python path or the same directory.")
    BatchedAgiSimEnv = None # Placeholder

# --- ✅ TODO: Configuration for Stabilization Policy Run ---
# Update this path to point to your saved stabilization policy
POLICY_DIR = "policies/ppo_policy_2838.556396484375"


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters')

# Define the number of drones to visualize
NUM_DRONES_TO_VISUALIZE = 10
NUM_EVAL_STEPS = 800

# Provide an empty list for the track layout to satisfy the environment constructor
TRACK_LAYOUT_FOR_POLICY_RUN = []

# Scaling bounds for position and velocity (ensure these match your environment's scaling)
OBS_POS_MIN_NP = np.array([-50.0, -50.0, -50.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([50.0, 50.0, 50.0], dtype=np.float32)

# Based on paper's randomization for stabilization task
OBS_VEL_MIN_NP = np.array([-40.0, -40.0, -40.0], dtype=np.float32)
OBS_VEL_MAX_NP = np.array([40.0, 40.0, 40.0], dtype=np.float32)

def unscale_vector(scaled_vec_np, min_bounds_np, max_bounds_np):
    """
    Unscales a vector from the [-1, 1] range back to original world coordinates.
    """
    range_bounds = max_bounds_np - min_bounds_np
    # Avoid division by zero if min and max are the same for a dimension
    for i in range(len(range_bounds)):
        if np.isclose(range_bounds[i], 0.0):
            range_bounds[i] = 1.0 # Prevent division by zero, effectively no scaling
    unscaled_vec = ((scaled_vec_np + 1.0) * range_bounds / 2.0) + min_bounds_np
    return unscaled_vec

def run_policy_and_collect_data(policy_dir, env_params, num_steps, num_drones):
    """
    Runs the loaded policy in the environment and collects trajectory, velocity,
    actions, observations, and rewards for each drone.
    """
    if BatchedAgiSimEnv is None:
        print("Error: BatchedAgiSimEnv is not available. Cannot run policy.")
        # FIX: Return 6 values on error
        return None, None, None, None, None, None

    print(f"Loading policy from: {policy_dir}")
    try:
        loaded_policy = tf.saved_model.load(policy_dir)
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        # FIX: Return 6 values on error
        return None, None, None, None, None, None

    print(f"Initializing environment for policy run with {num_drones} drones...")
    py_eval_env = None
    try:
        # FIX: Pass the empty track_layout to the constructor
        py_eval_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=num_drones,
            track_layout=TRACK_LAYOUT_FOR_POLICY_RUN
        )
        eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
        print(f"Environment initialized with batch size: {eval_env.batch_size}.")
        if eval_env.batch_size != num_drones:
            print(f"Warning: Environment batch size {eval_env.batch_size} does not match requested num_drones {num_drones}.")
            num_drones = eval_env.batch_size
    except Exception as e:
        print(f"Error initializing environment: {e}")
        if py_eval_env is not None and hasattr(py_eval_env, 'close'):
            py_eval_env.close()
        # FIX: Return 6 values on error
        return None, None, None, None, None, None


    py_eval_env.setEval()
    actual_num_drones_in_env = eval_env.batch_size

    trajectories_unscaled = [[] for _ in range(actual_num_drones_in_env)]
    velocities_unscaled = [[] for _ in range(actual_num_drones_in_env)]
    actions_raw = [[] for _ in range(actual_num_drones_in_env)]
    obs_shared_scaled = [[] for _ in range(actual_num_drones_in_env)]

    obs_task_scaled = [[] for _ in range(actual_num_drones_in_env)]


    rewards_data = [[] for _ in range(actual_num_drones_in_env)]
    total_rewards_accumulated = np.zeros(actual_num_drones_in_env, dtype=np.float32)

    active_drones = [True] * actual_num_drones_in_env
    time_step = eval_env.reset()
    policy_state = loaded_policy.get_initial_state(eval_env.batch_size)

    if 'shared_obs' not in time_step.observation:
        print("Error: Initial observation structure is incorrect. Missing 'shared_obs'.")
        if hasattr(eval_env, 'close'): eval_env.close()
        # FIX: Return 6 values on error
        return None, None, None, None, None, None

    # Process initial state
    initial_shared_obs_batch = time_step.observation['shared_obs'].numpy()
    initial_task_obs_batch = time_step.observation['task_specific_obs'].numpy()
    for i in range(actual_num_drones_in_env):
        obs_shared_scaled[i].append(initial_shared_obs_batch[i].copy())
        obs_task_scaled[i].append(initial_task_obs_batch[i].copy())

        # Unscale and store initial position
        scaled_pos = initial_shared_obs_batch[i, :3]
        unscaled_pos = unscale_vector(scaled_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP)
        trajectories_unscaled[i].append(unscaled_pos.tolist())

        # Unscale and store initial velocity
        scaled_vel = initial_shared_obs_batch[i, 9:12] # Based on obs structure [p, R, v, ...]
        unscaled_vel = unscale_vector(scaled_vel, OBS_VEL_MIN_NP, OBS_VEL_MAX_NP)
        velocities_unscaled[i].append(unscaled_vel.tolist())


    print(f"Running policy for up to {num_steps} steps for {actual_num_drones_in_env} drones...")
    try:
        for step_num in range(num_steps):
            if not np.any(active_drones):
                print(f"All drones finished before {num_steps} steps. Stopping at step {step_num}.")
                break

            action_step = loaded_policy.action(time_step, policy_state)
            current_actions_batch = action_step.action.numpy()
            time_step = eval_env.step(action_step.action)

            rewards_batch = time_step.reward.numpy()
            next_shared_obs_batch = time_step.observation['shared_obs'].numpy()
            next_task_obs_batch = time_step.observation['task_specific_obs'].numpy()
            is_last_batch = time_step.is_last().numpy()

            for i in range(actual_num_drones_in_env):
                if active_drones[i]:
                    actions_raw[i].append(current_actions_batch[i].copy())
                    rewards_data[i].append(rewards_batch[i])
                    total_rewards_accumulated[i] += rewards_batch[i]
                    obs_shared_scaled[i].append(next_shared_obs_batch[i].copy())
                    obs_task_scaled[i].append(next_task_obs_batch[i].copy())

                    # Unscale and store current position
                    scaled_pos = next_shared_obs_batch[i, :3]
                    unscaled_pos = unscale_vector(scaled_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP)
                    trajectories_unscaled[i].append(unscaled_pos.tolist())

                    # Unscale and store current velocity
                    scaled_vel = next_shared_obs_batch[i, 9:12]
                    unscaled_vel = unscale_vector(scaled_vel, OBS_VEL_MIN_NP, OBS_VEL_MAX_NP)
                    velocities_unscaled[i].append(unscaled_vel.tolist())

                    if is_last_batch[i]:
                        active_drones[i] = False
            policy_state = action_step.state
    finally:
        if hasattr(eval_env, 'close'):
            eval_env.close()
            print("Environment closed.")

    print("\nFinished policy run.")
    return (trajectories_unscaled,
            velocities_unscaled,
            [np.array(a) for a in actions_raw],
            [np.array(o) for o in obs_shared_scaled],
            [np.array(o) for o in obs_task_scaled], # NEW
            [np.array(r) for r in rewards_data],
            total_rewards_accumulated)

def visualize_stabilization_trajectories(trajectories_list=None, title="3D Stabilization Trajectories"):
    """
    Visualizes the 3D trajectories of drones without any track layout.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    all_x, all_y, all_z = [], [], []

    if trajectories_list:
        num_trajectories = len(trajectories_list)
        colors_map = plt.cm.get_cmap('viridis', num_trajectories if num_trajectories > 0 else 1)

        for idx, trajectory_points in enumerate(trajectories_list):
            if trajectory_points and len(trajectory_points) > 1:
                traj_np = np.array(trajectory_points)
                drone_color = colors_map(idx / num_trajectories) if num_trajectories > 1 else 'blue'
                ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], color=drone_color, linewidth=1.5, alpha=0.8, label=f'Drone {idx} Traj')
                all_x.extend(traj_np[:, 0]); all_y.extend(traj_np[:, 1]); all_z.extend(traj_np[:, 2])

                # Mark start and end points
                ax.scatter(traj_np[0,0], traj_np[0,1], traj_np[0,2], color=drone_color, s=100, marker='o', label=f'D{idx} Start', depthshade=False, zorder=10)
                ax.scatter(traj_np[-1,0], traj_np[-1,1], traj_np[-1,2], color=drone_color, s=100, marker='X', label=f'D{idx} End', depthshade=False, zorder=10)

    # Set plot limits and aspect ratio
    if not all_x:
        ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([0, 10])
    else:
        max_range = np.array([max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)]).max() / 2.0
        mid_x = (max(all_x)+min(all_x)) * 0.5
        mid_y = (max(all_y)+min(all_y)) * 0.5
        mid_z = (max(all_z)+min(all_z)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X-axis (m)'); ax.set_ylabel('Y-axis (m)'); ax.set_zlabel('Z-axis (m)')
    ax.set_title(title)

    plt.tight_layout()

def plot_attitude_history(shared_obs_np, title="Attitude (Euler Angles) Over Time", drone_index=0):
    """
    Calculates and plots the drone's attitude (roll, pitch, yaw) in degrees over time.
    """
    if shared_obs_np is None or shared_obs_np.ndim != 2 or shared_obs_np.shape[1] < 9:
        print(f"No or invalid shared observation data for attitude plot for drone {drone_index}.")
        return

    num_timesteps = shared_obs_np.shape[0]
    time_axis = np.arange(num_timesteps) * 0.01  # Assuming 100Hz sim rate

    euler_angles_list = []
    for i in range(num_timesteps):
        # Reconstruct the rotation matrix from the first two columns in the observation
        r_col0 = shared_obs_np[i, 3:6]
        r_col1 = shared_obs_np[i, 6:9]
        r_col2 = np.cross(r_col0, r_col1)
        R = np.column_stack([r_col0, r_col1, r_col2])

        # Calculate Euler angles from the rotation matrix
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        # Convert to degrees and append
        euler_angles_list.append(np.rad2deg([roll, pitch, yaw]))

    euler_angles_np = np.array(euler_angles_list)

    # Create the plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index})")

    labels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        axs[i].plot(time_axis, euler_angles_np[:, i], label=labels[i], color=colors[i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_acceleration_history(task_obs_np, title="Scaled Acceleration Over Time", drone_index=0):
    """
    Plots the scaled acceleration components (ax, ay, az) over time.
    """
    if task_obs_np is None or not isinstance(task_obs_np, np.ndarray) or task_obs_np.ndim != 2 or task_obs_np.shape[1] < 3:
        print(f"No or invalid task-specific observation data to plot for drone {drone_index}.")
        return

    # We use num_timesteps from the task_obs, which includes the initial state
    num_timesteps = task_obs_np.shape[0]
    time_axis = np.arange(num_timesteps) * 0.01 # Assuming 100Hz sim rate for time axis

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index})")

    labels = ['Accel X (scaled)', 'Accel Y (scaled)', 'Accel Z (scaled)']
    colors = ['#ff7f0e', '#2ca02c', '#d62728'] # Using different colors

    for i in range(3):
        # We plot the acceleration from the task-specific observation
        axs[i].plot(time_axis, task_obs_np[:, i], label=labels[i], color=colors[i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
        axs[i].set_ylim(-1.1, 1.1)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_velocity_history(velocities_np, title="Unscaled Velocity Over Time", drone_index=0):
    """
    Plots the unscaled velocity components (x, y, z) over time for a single drone.
    """
    if velocities_np is None or not isinstance(velocities_np, np.ndarray) or velocities_np.ndim != 2 or velocities_np.shape[1] < 3:
        return

    num_timesteps, _ = velocities_np.shape
    time_axis = np.arange(num_timesteps) * 0.01 # Assuming 100Hz sim rate

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index})")

    labels = ['Velocity X (m/s)', 'Velocity Y (m/s)', 'Velocity Z (m/s)']
    colors = ['r', 'g', 'b']

    for i in range(3):
        axs[i].plot(time_axis, velocities_np[:, i], label=labels[i], color=colors[i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_actions_history(actions_np, title="Policy Raw Actions Over Time", drone_index=0):
    """
    Plots the normalized actions over time.
    """
    if actions_np is None or actions_np.ndim != 2 or actions_np.shape[0] == 0 or actions_np.shape[1] < 4:
        return

    num_timesteps, num_action_dims = actions_np.shape
    time_axis = np.arange(num_timesteps)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index})")

    action_labels = ['Thrust (norm)', 'Rate X (norm)', 'Rate Y (norm)', 'Rate Z (norm)']
    colors = ['r', 'g', 'b', 'purple']

    for i in range(4):
        axs[i].plot(time_axis, actions_np[:, i], label=action_labels[i], color=colors[i], marker='.', linestyle='-')
        axs[i].set_ylabel(action_labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
        axs[i].set_ylim(-1.1, 1.1)

    axs[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_rewards_history(rewards_np, total_reward, title="Step Rewards Over Time", drone_index=0):
    """
    Plots the reward per step over time.
    """
    if rewards_np is None or rewards_np.size == 0:
        return

    plt.figure(figsize=(12, 6))
    plt.ylim([-0.0752, 0.001])
    plt.plot(rewards_np, label=f'Drone {drone_index} Reward/Step', color='orange', marker='.', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title(f"{title} (Drone {drone_index})\nTotal Accumulated Reward: {total_reward:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_shared_observations_history(shared_obs_np, title="Scaled Shared Observations", drone_index=0):
    """
    Plots the shared part of the observation vector over time.
    """
    if shared_obs_np is None or shared_obs_np.shape[0] <= 1 or shared_obs_np.shape[1] < 15:
        return

    num_timesteps = shared_obs_np.shape[0]
    time_axis = np.arange(num_timesteps)

    fig, axs = plt.subplots(5, 1, figsize=(15, 18), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index})")

    # Plot scaled positions
    axs[0].plot(time_axis, shared_obs_np[:, 0], label='Pos X (s)'); axs[0].plot(time_axis, shared_obs_np[:, 1], label='Pos Y (s)'); axs[0].plot(time_axis, shared_obs_np[:, 2], label='Pos Z (s)')
    axs[0].set_ylabel('Position (scaled)'); axs[0].legend(); axs[0].grid(True); axs[0].set_ylim(-1.1, 1.1)

    # Plot scaled velocities
    axs[1].plot(time_axis, shared_obs_np[:, 9], label='Vel X (s)'); axs[1].plot(time_axis, shared_obs_np[:, 10], label='Vel Y (s)'); axs[1].plot(time_axis, shared_obs_np[:, 11], label='Vel Z (s)')
    axs[1].set_ylabel('Velocity (scaled)'); axs[1].legend(); axs[1].grid(True); axs[1].set_ylim(-1.1, 1.1)

    # Plot scaled angular velocities
    axs[2].plot(time_axis, shared_obs_np[:, 12], label='AngVel X (s)'); axs[2].plot(time_axis, shared_obs_np[:, 13], label='AngVel Y (s)'); axs[2].plot(time_axis, shared_obs_np[:, 14], label='AngVel Z (s)')
    axs[2].set_ylabel('AngVel (scaled)'); axs[2].legend(); axs[2].grid(True); axs[2].set_ylim(-1.1, 1.1)

    # Plot rotation matrix components (example: first column)
    axs[3].plot(time_axis, shared_obs_np[:, 3], label='R_wb[0,0]'); axs[3].plot(time_axis, shared_obs_np[:, 4], label='R_wb[1,0]'); axs[3].plot(time_axis, shared_obs_np[:, 5], label='R_wb[2,0]')
    axs[3].set_ylabel('R_wb col0'); axs[3].legend(); axs[3].grid(True); axs[3].set_ylim(-1.1, 1.1)

    # Plot previous actions
    axs[4].plot(time_axis, shared_obs_np[:, 15], label='PrevAct Thrust'); axs[4].plot(time_axis, shared_obs_np[:, 16], label='PrevAct RateX');
    axs[4].set_ylabel('Prev Action (Obs)'); axs[4].legend(); axs[4].grid(True); axs[4].set_ylim(-1.1, 1.1)

    axs[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def main():
    if not os.path.isdir(POLICY_DIR):
        print(f"Error: Policy directory '{POLICY_DIR}' not found. Please update the path.")
        return

    env_params = {
        'sim_config_path': SIM_CONFIG_PATH,
        'agi_param_dir': AGI_PARAM_DIR,
        'sim_base_dir': SIM_BASE_DIR,
    }

    # Run the policy and collect all data
    (trajectories, velocities, actions,
     shared_obs, task_obs, rewards, total_rewards) = run_policy_and_collect_data(
        POLICY_DIR, env_params, NUM_EVAL_STEPS, NUM_DRONES_TO_VISUALIZE
    )

    if trajectories is None:
        print("Policy run failed. Exiting.")
        return

    # Visualize the 3D trajectories for all drones
    visualize_stabilization_trajectories(
        trajectories_list=trajectories,
        title=f"{NUM_DRONES_TO_VISUALIZE} Drone(s) Stabilization Trajectories"
    )

    # Find the best drone to plot detailed data for (highest reward)
    if total_rewards is not None and total_rewards.size > 0:
        best_drone_idx = np.argmax(total_rewards)
        print(f"\n--- Generating 2D plots for Drone {best_drone_idx} (Highest Reward: {total_rewards[best_drone_idx]:.4f}) ---")

        plot_acceleration_history(
            np.array(task_obs[best_drone_idx]),
            title="Scaled Acceleration vs. Time",
            drone_index=best_drone_idx
        )

        # Plot velocities for the best drone
        plot_velocity_history(
            np.array(velocities[best_drone_idx]),
            title="Unscaled Velocity vs. Time",
            drone_index=best_drone_idx
        )

        plot_attitude_history(
            shared_obs[best_drone_idx],
            title="Attitude vs. Time",
            drone_index=best_drone_idx
        )

        # Plot actions for the best drone
        plot_actions_history(
            actions[best_drone_idx],
            title="Policy Raw Actions",
            drone_index=best_drone_idx
        )

        # Plot observations for the best drone
        plot_shared_observations_history(
            shared_obs[best_drone_idx],
            title="Scaled Shared Observations",
            drone_index=best_drone_idx
        )

        # Plot rewards for the best drone
        plot_rewards_history(
            rewards[best_drone_idx],
            total_rewards[best_drone_idx],
            title="Step Rewards",
            drone_index=best_drone_idx
        )

    # Show all plots
    if plt.get_fignums():
        print(f"\nDisplaying {len(plt.get_fignums())} plot window(s)...")
        plt.show()
    else:
        print("\nNo plots were generated.")


if __name__ == "__main__":
    main()