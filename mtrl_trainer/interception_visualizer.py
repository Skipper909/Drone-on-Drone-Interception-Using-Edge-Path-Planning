import os  # For path joining

# --- TF-Agents and TensorFlow Imports ---
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import json

from mtrl_lib.gate_utils import load_track_from_file

try:
    # Ensure this path is correct or the module is in PYTHONPATH
    from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv  # As used in your training script
    # from auto_reset_wrapper import AutoResetWrapper   # As used in your training script
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv or AutoResetWrapper.")
    print("Please ensure these files are in the Python path or the same directory.")
    BatchedAgiSimEnv = None

# --- Configuration for Policy Run ---
POLICY_DIR = "policies/ppo_policy_end_of_run_new"  # Directory where your trained policy is saved
#POLICY_DIR = "policies/ppo_policy_mtrl"  # Directory where your trained policy is saved

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

NUM_DRONES_TO_VISUALIZE = 1
NUM_EVAL_STEPS = 25000  # Number of steps to run the policy for trajectory collection

# Scaling bounds for position (ensure these match your environment's scaling)
OBS_POS_MIN_NP = np.array([-200.0, -200.0, -200.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([200.0, 200.0, 200.0], dtype=np.float32)


def unscale_position(scaled_pos_np, min_bounds_np, max_bounds_np):
    """
    Unscales a 3D position vector from the [-1, 1] range back to original world coordinates.
    """
    range_bounds = max_bounds_np - min_bounds_np
    # Avoid division by zero if min and max are the same for a dimension
    for i in range(len(range_bounds)):
        if np.isclose(range_bounds[i], 0.0):
            range_bounds[i] = 1.0  # Prevent division by zero, effectively no scaling for this dim
    unscaled_pos = ((scaled_pos_np + 1.0) * range_bounds / 2.0) + min_bounds_np
    return unscaled_pos


def run_policy_and_collect_trajectory(policy_dir, env_params, num_steps, num_drones):
    """
    Runs the loaded policy in the environment and collects trajectory, actions,
    observations, and rewards for each drone independently.
    """
    if BatchedAgiSimEnv is None:
        print("Error: BatchedAgiSimEnv is not available. Cannot run policy.")
        return None, None, None, None, None, None  # trajectories, actions, shared_obs, task_obs, rewards, total_rewards

    print(f"Loading policy from: {policy_dir}")
    try:
        loaded_policy = tf.saved_model.load(policy_dir)
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None, None, None, None, None, None

    print(f"Initializing environment for policy run with {num_drones} drones...")
    py_eval_env = None  # Define to ensure it's available in finally block if init fails

    final_track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    try:
        # Ensure your BatchedAgiSimEnv can be initialized with num_drones
        py_eval_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=num_drones,
            track_layout=final_track_layout  # Pass track layout if used by env
        )
        eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
        print(f"Environment initialized with batch size: {eval_env.batch_size}.")
        if eval_env.batch_size != num_drones:
            print(f"Warning: Environment batch size {eval_env.batch_size} does not match requested num_drones {num_drones}. Adjusting internal drone count.")
            # The environment itself dictates the actual number of drones if it's fixed internally
            num_drones = eval_env.batch_size
    except Exception as e:
        print(f"Error initializing environment: {e}")
        if py_eval_env is not None and hasattr(py_eval_env, 'close'):  # Attempt to close if partially initialized
            py_eval_env.close()
        return None, None, None, None, None, None

    # Get the actual underlying PyEnvironment (for setEval, setNoiseIntensity, etc.)
    actual_py_env_interface = None
    if hasattr(py_eval_env, '_env'):
        actual_py_env_interface = py_eval_env._env
    elif hasattr(py_eval_env, 'environment'):
        actual_py_env_interface = py_eval_env.environment
    else:
        actual_py_env_interface = py_eval_env

    # Optional: load previous "gate pass states" file if you're using that reset logic
    json_file_path = os.path.join(policy_dir, "successful_gate_pass_states.json")

    if os.path.exists(json_file_path):
        try:
            print(f"Loading gate pass states from {json_file_path}...")
            with open(json_file_path, 'r') as f:
                loaded_gate_pass_data = json.load(f)

            # If your env supports it, you can call a setter here, e.g.:
            # actual_py_env_interface.set_successful_gate_pass_states_data(loaded_gate_pass_data)
            print("Successfully loaded gate pass states (not applied unless env supports it).")
        except Exception as e:
            print(f"Error loading or setting gate pass states: {e}")
    else:
        print(f"Warning: Gate pass states file not found at {json_file_path}. Evaluation will use empty/default initial pass states.")

    # Make sure we call control methods on the underlying PyEnvironment (not TFPyEnvironment)
    if hasattr(actual_py_env_interface, "setEval"):
        actual_py_env_interface.setEval()
    else:
        print("Warning: underlying env has no setEval() method.")

    if hasattr(actual_py_env_interface, "setNoiseIntensity"):
        actual_py_env_interface.setNoiseIntensity(
            vio_pos_drift_std=0,
            vio_att_drift_std_deg=0,
            gate_reset_pos_std=0,
            gate_reset_att_std_deg=0
        )
    else:
        print("Warning: underlying env has no setNoiseIntensity() method.")

    actual_num_drones_in_env = eval_env.batch_size  # This should be NUM_DRONES_TO_VISUALIZE (or adjusted)

    trajectories_unscaled_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    actions_raw_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    obs_shared_scaled_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    obs_task_scaled_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    rewards_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    total_rewards_accumulated_all_drones = np.zeros(actual_num_drones_in_env, dtype=np.float32)

    active_drones = [True] * actual_num_drones_in_env
    time_step = eval_env.reset()
    policy_state = loaded_policy.get_initial_state(eval_env.batch_size)

    if 'shared_obs' not in time_step.observation or \
            'task_specific_obs' not in time_step.observation:
        print("Error: Initial observation structure is incorrect. Missing 'shared_obs' or 'task_specific_obs'.")
        if hasattr(eval_env, 'close'):
            eval_env.close()
        return None, None, None, None, None, None

    initial_shared_obs_batch = time_step.observation['shared_obs'].numpy()
    initial_task_obs_batch = time_step.observation['task_specific_obs'].numpy()

    for i in range(actual_num_drones_in_env):
        obs_shared_scaled_all_drones[i].append(initial_shared_obs_batch[i].copy())
        obs_task_scaled_all_drones[i].append(initial_task_obs_batch[i].copy())
        scaled_start_position = initial_shared_obs_batch[i, :3]
        unscaled_start_position = unscale_position(scaled_start_position, OBS_POS_MIN_NP, OBS_POS_MAX_NP)
        trajectories_unscaled_all_drones[i].append(unscaled_start_position.tolist())

    print(f"Running policy for up to {num_steps} steps for {actual_num_drones_in_env} drones...")
    final_step_num = 0
    try:
        for step_num in range(num_steps):
            final_step_num = step_num
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
                    actions_raw_all_drones[i].append(current_actions_batch[i].copy())

                    current_reward_drone_i = rewards_batch[i]
                    rewards_all_drones[i].append(current_reward_drone_i)
                    total_rewards_accumulated_all_drones[i] += current_reward_drone_i

                    obs_shared_scaled_all_drones[i].append(next_shared_obs_batch[i].copy())
                    obs_task_scaled_all_drones[i].append(next_task_obs_batch[i].copy())

                    scaled_current_position = next_shared_obs_batch[i, :3]
                    unscaled_current_position = unscale_position(
                        scaled_current_position,
                        OBS_POS_MIN_NP,
                        OBS_POS_MAX_NP
                    )
                    trajectories_unscaled_all_drones[i].append(unscaled_current_position.tolist())

                    if is_last_batch[i]:
                        active_drones[i] = False
                        print(f"Drone {i} finished at environment step {step_num + 1} "
                              f"(Recorded {len(actions_raw_all_drones[i])} actions for this drone).")
            policy_state = action_step.state
    finally:  # Ensure environment is closed
        if hasattr(eval_env, 'close'):
            eval_env.close()
            print("TFPyEnvironment closed.")
        elif py_eval_env is not None and hasattr(py_eval_env, 'close'):  # If TFPyEnv wrapper failed but py_env was made
            py_eval_env.close()
            print("PyEnvironment closed.")

    print(f"\nFinished policy run after {final_step_num + 1} potential environment steps.")
    print("Collected data summary:")
    for i in range(actual_num_drones_in_env):
        print(f"  Drone {i}: Ran for {len(actions_raw_all_drones[i])} action steps. "
              f"Total Reward: {total_rewards_accumulated_all_drones[i]:.4f}. "
              f"Trajectory points: {len(trajectories_unscaled_all_drones[i])}")

    actions_final = [np.array(actions) if actions else np.array([]) for actions in actions_raw_all_drones]
    obs_shared_final = [np.array(obs) if obs else np.array([]) for obs in obs_shared_scaled_all_drones]
    obs_task_final = [np.array(obs) if obs else np.array([]) for obs in obs_task_scaled_all_drones]
    rewards_final = [np.array(rewards) if rewards else np.array([]) for rewards in rewards_all_drones]

    eval_env.reset()

    return (trajectories_unscaled_all_drones,
            actions_final,
            obs_shared_final,
            obs_task_final,
            rewards_final,
            total_rewards_accumulated_all_drones)


def plot_rewards_history(rewards_np, total_reward, title="Step Rewards Over Time", drone_index=0):
    if rewards_np is None or not isinstance(rewards_np, np.ndarray) or rewards_np.ndim == 0 or rewards_np.shape[0] == 0:
        return
    num_timesteps = rewards_np.shape[0]
    time_axis = np.arange(num_timesteps)
    plt.figure(figsize=(12, 6))
    plt.ylim([-0.042, 0.042])
    plt.plot(time_axis, rewards_np, label=f'Drone {drone_index} Reward/Step',
             color='orange', marker='.', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title(f"{title} (Drone {drone_index}, {num_timesteps} steps)\n"
              f"Total Accumulated Reward: {total_reward:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def visualize_track_layout(track_layout, trajectories_list=None,
                           title="3D Interception Scenario", normal_vector_length=0.5):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []

    for i, gate_corners_list in enumerate(track_layout):
        gate_corners_np = np.array(gate_corners_list)
        gate_polygon = Poly3DCollection([gate_corners_np], alpha=0.2,
                                        linewidths=1, edgecolors='k')
        gate_colors = ['lightblue', 'lightgreen', 'lightcoral',
                       'lightgoldenrodyellow', 'plum', 'lightpink']
        gate_polygon.set_facecolor(gate_colors[i % len(gate_colors)])
        ax.add_collection3d(gate_polygon)

        center_point = np.mean(gate_corners_np, axis=0)
        ax.text(center_point[0], center_point[1], center_point[2],
                f" G{i}", color='k', fontsize=8)

        v1 = gate_corners_np[1] - gate_corners_np[0]
        v2 = gate_corners_np[3] - gate_corners_np[0]
        normal_vector = np.cross(v1, v2)
        norm_magnitude = np.linalg.norm(normal_vector)
        if norm_magnitude > 1e-6:
            normal_vector /= norm_magnitude
        else:
            normal_vector = np.array([0, 0, 1])
        ax.quiver(center_point[0], center_point[1], center_point[2],
                  normal_vector[0], normal_vector[1], normal_vector[2],
                  length=normal_vector_length, color='darkmagenta',
                  arrow_length_ratio=0.3,
                  label='Gate Normal' if i == 0 else "")

        all_x.extend(gate_corners_np[:, 0])
        all_y.extend(gate_corners_np[:, 1])
        all_z.extend(gate_corners_np[:, 2])

    if trajectories_list:
        num_trajectories = len(trajectories_list)
        try:
            colors_map = plt.cm.get_cmap('viridis', num_trajectories if num_trajectories > 0 else 1)
        except Exception:
            colors_map = plt.cm.get_cmap('hsv', num_trajectories if num_trajectories > 0 else 1)

        for idx, trajectory_points in enumerate(trajectories_list):
            if trajectory_points and len(trajectory_points) > 1:
                traj_np = np.array(trajectory_points)
                num_steps_for_traj = len(traj_np) - 1
                drone_color = colors_map(idx / num_trajectories) if num_trajectories > 1 else 'blue'
                ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2],
                        color=drone_color, linewidth=1,
                        label=f'Drone {idx} Traj ({num_steps_for_traj} steps)',
                        marker='.', markersize=2)
                all_x.extend(traj_np[:, 0])
                all_y.extend(traj_np[:, 1])
                all_z.extend(traj_np[:, 2])
                ax.scatter(traj_np[0, 0], traj_np[0, 1], traj_np[0, 2],
                           color=drone_color, s=80, marker='o',
                           label=f'D{idx} Start', depthshade=False, zorder=10)
                ax.scatter(traj_np[-1, 0], traj_np[-1, 1], traj_np[-1, 2],
                           color=drone_color, s=80, marker='X',
                           label=f'D{idx} End', depthshade=False, zorder=10)

    # Origin / spawn reference at (0,0,1)
    dot_x, dot_y, dot_z = 0, 0, 1
    ax.scatter([dot_x], [dot_y], [dot_z], color='red', s=60,
               marker='^', label='Origin Ref (0,0,1)')
    all_x.append(dot_x)
    all_y.append(dot_y)
    all_z.append(dot_z)

    if not all_x:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 10])
    else:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)

        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
        ax.set_zlim(min(min_z - 0.5, 0) if min_z < 0.5 else min_z - 0.5,
                    max_z + 0.5)

    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Z-axis (m)')
    ax.set_title(title)

    # Keep legend dedup logic simple
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), loc='upper left',
              bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-3])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    plt.tight_layout(rect=[0, 0, 0.82, 1])


def plot_actions_history(actions_np, title="Policy Raw Actions Over Time", drone_index=0):
    if actions_np is None or not isinstance(actions_np, np.ndarray) or actions_np.ndim != 2 or actions_np.shape[0] == 0:
        return
    num_timesteps, num_action_dims = actions_np.shape
    if num_action_dims < 4:
        return
    time_axis = np.arange(num_timesteps)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index}, {num_timesteps} steps)")
    action_labels = ['Thrust (norm)', 'Rate X (norm)', 'Rate Y (norm)', 'Rate Z (norm)']
    colors = ['r', 'g', 'b', 'purple']
    for i in range(4):
        axs[i].plot(time_axis, actions_np[:, i], label=action_labels[i],
                    color=colors[i], marker='.', linestyle='-')
        axs[i].set_ylabel(action_labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
        axs[i].set_ylim(-1.1, 1.1)
    axs[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_observations_history(shared_obs_np, task_specific_obs_np,
                              title="Scaled Observations Over Time",
                              drone_index=0):
    """
    For interception:
      shared_obs: same as before (positions, R_wb cols, v, w, prev_action...)
      task_specific_obs:
        0:3 -> p_rel (target - drone)
        3:6 -> v_ego
        6   -> dist
        7   -> vmag
    """
    if (shared_obs_np is None or task_specific_obs_np is None or
            not isinstance(shared_obs_np, np.ndarray) or
            not isinstance(task_specific_obs_np, np.ndarray) or
            shared_obs_np.shape[0] <= 1 or
            task_specific_obs_np.shape[0] <= 1):
        return

    num_obs_timesteps = shared_obs_np.shape[0]
    obs_time_axis = np.arange(num_obs_timesteps)

    num_shared_plots = 5  # pos, Rcol0, Rcol1, vel, angvel+prev_action
    num_task_plots = 3    # p_rel, v_ego, dist/vmag
    total_plots = num_shared_plots + num_task_plots

    fig, axs = plt.subplots(total_plots, 1, figsize=(15, 22), sharex=True)
    fig.suptitle(f"{title} (Drone {drone_index}, {num_obs_timesteps-1} steps after init)")
    plot_idx = 0

    # Shared obs: as in the racing visualizer (assuming at least 19 dims)
    if shared_obs_np.shape[1] < 19:
        print(f"Shared observations for drone {drone_index} have {shared_obs_np.shape[1]} "
              f"columns, expected at least 19.")
        for _ in range(num_shared_plots):
            axs[plot_idx].text(0.5, 0.5, 'Not enough shared_obs data columns',
                               ha='center', va='center', color='red')
            axs[plot_idx].set_xticks([])
            axs[plot_idx].set_yticks([])
            plot_idx += 1
    else:
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 0], label='Pos X (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 1], label='Pos Y (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 2], label='Pos Z (s)')
        axs[plot_idx].set_ylabel('Position (s)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        axs[plot_idx].set_ylim(-1.1, 1.1)
        plot_idx += 1

        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 3], label='R_wb[0,0]')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 4], label='R_wb[1,0]')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 5], label='R_wb[2,0]')
        axs[plot_idx].set_ylabel('R_wb col0')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        axs[plot_idx].set_ylim(-1.1, 1.1)
        plot_idx += 1

        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 6], label='R_wb[0,1]')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 7], label='R_wb[1,1]')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 8], label='R_wb[2,1]')
        axs[plot_idx].set_ylabel('R_wb col1')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        axs[plot_idx].set_ylim(-1.1, 1.1)
        plot_idx += 1

        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 9], label='Vel X (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 10], label='Vel Y (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 11], label='Vel Z (s)')
        axs[plot_idx].set_ylabel('Velocity (s)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        axs[plot_idx].set_ylim(-1.1, 1.1)
        plot_idx += 1

        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 12], label='AngVel X (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 13], label='AngVel Y (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 14], label='AngVel Z (s)')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 15], label='PrevAct Thrust')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 16], label='PrevAct RateX')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 17], label='PrevAct RateY')
        axs[plot_idx].plot(obs_time_axis, shared_obs_np[:, 18], label='PrevAct RateZ')
        axs[plot_idx].set_ylabel('AngVel & PrevAct (s)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        axs[plot_idx].set_ylim(-1.1, 1.1)
        plot_idx += 1

    # Task-specific obs for interception
    if task_specific_obs_np.shape[1] >= 8:
        # p_rel = target - drone
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 0], label='p_rel X')
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 1], label='p_rel Y')
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 2], label='p_rel Z')
        axs[plot_idx].set_ylabel('p_rel (scaled)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

        # v_ego
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 3], label='v_ego X')
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 4], label='v_ego Y')
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 5], label='v_ego Z')
        axs[plot_idx].set_ylabel('v_ego (scaled)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

        # dist & vmag
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 6], label='dist')
        axs[plot_idx].plot(obs_time_axis, task_specific_obs_np[:, 7], label='|v_ego|')
        axs[plot_idx].set_ylabel('dist / vmag (scaled)')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1
    else:
        # Not enough dims for our interception structure
        for _ in range(num_task_plots):
            axs[plot_idx].text(0.5, 0.5, 'Not enough task_specific_obs for interception plot',
                               ha='center', va='center', color='red')
            axs[plot_idx].set_xticks([])
            axs[plot_idx].set_yticks([])
            plot_idx += 1

    axs[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.97])


def main():
    final_track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    if not os.path.isdir(POLICY_DIR):
        print(f"Warning: Policy directory '{POLICY_DIR}' not found. Policy run will likely fail.")

    env_params_for_run = {
        'sim_config_path': SIM_CONFIG_PATH if os.path.exists(SIM_CONFIG_PATH) else "dummy_sim_config.yaml",
        'agi_param_dir': AGI_PARAM_DIR if os.path.exists(AGI_PARAM_DIR) else "dummy_agi_params",
        'sim_base_dir': SIM_BASE_DIR if os.path.exists(SIM_BASE_DIR) else "dummy_sim_base",
    }

    # Create dummy files/dirs if they don't exist and are placeholders
    for p_key, p_val in env_params_for_run.items():
        if "dummy" in p_val:
            if "." in os.path.basename(p_val):  # It's a file
                if not os.path.exists(p_val):
                    print(f"Creating dummy file: {p_val}")
                    open(p_val, 'w').close()
            else:  # It's a directory
                if not os.path.exists(p_val):
                    print(f"Creating dummy directory: {p_val}")
                    os.makedirs(p_val, exist_ok=True)

    all_drone_trajectories = None
    all_raw_actions = None
    all_scaled_shared_obs = None
    all_scaled_task_obs = None
    all_rewards_data = None
    all_total_rewards = None

    policy_run_attempted = False
    if not os.path.exists(POLICY_DIR) or not os.listdir(POLICY_DIR):
        print(f"Warning: Policy directory '{POLICY_DIR}' is empty or does not exist. Skipping policy run.")
    elif BatchedAgiSimEnv is None:
        print(f"Warning: BatchedAgiSimEnv not available. Cannot run policy.")
    else:
        policy_run_attempted = True
        print(f"Attempting to run policy for {NUM_DRONES_TO_VISUALIZE} drone(s).")
        (all_drone_trajectories, all_raw_actions,
         all_scaled_shared_obs, all_scaled_task_obs,
         all_rewards_data, all_total_rewards) = run_policy_and_collect_trajectory(
            POLICY_DIR, env_params_for_run, NUM_EVAL_STEPS, NUM_DRONES_TO_VISUALIZE
        )

    # Visualize the 3D track and all drone trajectories
    visualize_track_layout(
        final_track_layout,
        trajectories_list=all_drone_trajectories,
        title=f"{NUM_DRONES_TO_VISUALIZE if policy_run_attempted and all_drone_trajectories else 0} "
              f"Drone(s) Interception Trajectories"
    )

    # --- Plot detailed 2D data for the drone with the highest reward ---
    drone_to_plot_details = -1
    highest_reward_value_for_plot = -float('inf')
    valid_data_package = False

    if (policy_run_attempted and all_total_rewards is not None and
            isinstance(all_total_rewards, np.ndarray) and
            all_raw_actions is not None and isinstance(all_raw_actions, list) and
            all_scaled_shared_obs is not None and isinstance(all_scaled_shared_obs, list) and
            all_scaled_task_obs is not None and isinstance(all_scaled_task_obs, list) and
            all_rewards_data is not None and isinstance(all_rewards_data, list)):

        num_drones_from_rewards_array = all_total_rewards.size

        if (num_drones_from_rewards_array > 0 and
                len(all_raw_actions) == num_drones_from_rewards_array and
                len(all_scaled_shared_obs) == num_drones_from_rewards_array and
                len(all_scaled_task_obs) == num_drones_from_rewards_array and
                len(all_rewards_data) == num_drones_from_rewards_array):
            valid_data_package = True
        else:
            print("\nWarning: Data lists from policy run have inconsistent lengths or "
                  "total_rewards is empty/malformed. Skipping 2D plots.")
            print(f"  TotalRewards size: {all_total_rewards.size if all_total_rewards is not None else 'None'}, "
                  f"Num RawActions entries: {len(all_raw_actions) if all_raw_actions is not None else 'None'}")

    if valid_data_package:
        best_drone_index = np.argmax(all_total_rewards)

        def has_sufficient_data(idx):
            return (all_raw_actions[idx] is not None and all_raw_actions[idx].size > 0 and
                    all_scaled_shared_obs[idx] is not None and all_scaled_shared_obs[idx].shape[0] > 1 and
                    all_rewards_data[idx] is not None and all_rewards_data[idx].size > 0)

        if has_sufficient_data(best_drone_index):
            drone_to_plot_details = best_drone_index
            highest_reward_value_for_plot = all_total_rewards[best_drone_index]
            print(f"\n--- Generating 2D plots for Drone {drone_to_plot_details} "
                  f"(Highest Reward: {highest_reward_value_for_plot:.4f}) ---")
        else:
            print(f"\nDrone {best_drone_index} (highest reward: "
                  f"{all_total_rewards[best_drone_index]:.4f}) has insufficient data "
                  f"for detailed 2D plots.")
            sorted_reward_indices = np.argsort(all_total_rewards)[::-1]
            found_alternative = False
            for current_idx in sorted_reward_indices:
                if current_idx == best_drone_index:
                    continue
                if has_sufficient_data(current_idx):
                    drone_to_plot_details = current_idx
                    highest_reward_value_for_plot = all_total_rewards[current_idx]
                    print(f"--- Selecting alternative Drone {drone_to_plot_details} "
                          f"(Reward: {highest_reward_value_for_plot:.4f}) for 2D plots. ---")
                    found_alternative = True
                    break
            if not found_alternative:
                print("--- No suitable drone with sufficient data found for 2D plots after checking alternatives. ---")

    elif policy_run_attempted:
        print("\n--- Skipping 2D plots as data from policy run is incomplete, inconsistent, or empty. ---")
    else:
        print("\n--- Skipping 2D plots as policy run was not performed. ---")

    if drone_to_plot_details != -1:
        plot_title_suffix = f"(Drone {drone_to_plot_details}"
        if highest_reward_value_for_plot > -float('inf'):
            plot_title_suffix += f" - Reward: {highest_reward_value_for_plot:.2f})"
        else:
            plot_title_suffix += ")"

        plot_actions_history(all_raw_actions[drone_to_plot_details],
                             title=f"Policy Raw Actions {plot_title_suffix}",
                             drone_index=drone_to_plot_details)

        plot_observations_history(all_scaled_shared_obs[drone_to_plot_details],
                                  all_scaled_task_obs[drone_to_plot_details],
                                  title=f"Scaled Observations {plot_title_suffix}",
                                  drone_index=drone_to_plot_details)

        plot_rewards_history(all_rewards_data[drone_to_plot_details],
                             all_total_rewards[drone_to_plot_details],
                             title=f"Step Rewards {plot_title_suffix}",
                             drone_index=drone_to_plot_details)
    else:
        if valid_data_package:
            print("  No drone met criteria for detailed 2D plotting.")

    # Show all generated plots
    if plt.get_fignums():
        print(f"\nDisplaying {len(plt.get_fignums())} plot window(s)...")
        plt.show()
    else:
        print("\nNo plots were generated to display.")


if __name__ == "__main__":
    # This is a basic check to ensure the script can run even if paths are not set up
    # For actual execution, ensure POLICY_DIR, SIM_CONFIG_PATH etc. are correct.
    if not os.path.exists(POLICY_DIR):
        print(f"INFO: POLICY_DIR '{POLICY_DIR}' does not exist. Creating for test run.")
        os.makedirs(POLICY_DIR, exist_ok=True)
        # Create a dummy saved_model.pb so os.listdir(POLICY_DIR) is not empty
        if not os.path.exists(os.path.join(POLICY_DIR, "saved_model.pb")):
            open(os.path.join(POLICY_DIR, "saved_model.pb"), 'w').close()
    main()
