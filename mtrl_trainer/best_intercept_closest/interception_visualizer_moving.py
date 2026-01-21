import os  # For path joining

# --- TF-Agents and TensorFlow Imports ---
import tensorflow as tf
from tf_agents.environments import tf_py_environment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import json

from mtrl_lib.gate_utils import load_track_from_file

try:
    # Ensure this path is correct or the module is in PYTHONPATH
    from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv  # As used in your training script
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv.")
    print("Please ensure these files are in the Python path or the same directory.")
    BatchedAgiSimEnv = None

# --- Configuration for Policy Run ---
POLICY_DIR = "policies_intercept/best_intercept"  # Path to the saved policy directory

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

NUM_DRONES_TO_VISUALIZE = 1
NUM_EVAL_STEPS = 25000  # Number of steps to run the policy for trajectory collection

# Scaling bounds for position (must match environment scaling)
OBS_POS_MIN_NP = np.array([-200.0, -200.0, -200.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([200.0, 200.0, 200.0], dtype=np.float32)
TARGET_DT = 0.02  # seconds, must match simulator dt
TARGET_VEL_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, vz] m/s in world frame

# --- Capture radius for visualizer (must match INTERCEPTION_CAPTURE_RADIUS in reward_params.yaml) ---
CAPTURE_RADIUS_M = 2.0   # or 8.0 / 10.0 if you increased it in the env


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
    observations, rewards, and reconstructed target trajectories for each drone.

    Assumes the interception task-specific observation layout is:

        0:3 -> p_rel (target - drone) in world coords (unscaled, meters)
        3:6 -> v_ego (drone velocity in world)
        6   -> dist
        7   -> |v_ego|

    We still record p_rel etc for debugging/plots, but we'll reconstruct target
    trajectories analytically from gate center + TARGET_VEL_WORLD in main().
    """
    if BatchedAgiSimEnv is None:
        print("Error: BatchedAgiSimEnv is not available. Cannot run policy.")
        return None, None, None, None, None, None, None

    print(f"Loading policy from: {policy_dir}")
    try:
        loaded_policy = tf.saved_model.load(policy_dir)
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None, None, None, None, None, None, None

    print(f"Initializing environment for policy run with {num_drones} drones...")
    py_eval_env = None

    final_track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    try:
        py_eval_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=num_drones,
            track_layout=final_track_layout
        )
        eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
        print(f"Environment initialized with batch size: {eval_env.batch_size}.")
        if eval_env.batch_size != num_drones:
            print(f"Warning: Environment batch size {eval_env.batch_size} does not match requested num_drones {num_drones}. Adjusting internal drone count.")
            num_drones = eval_env.batch_size
    except Exception as e:
        print(f"Error initializing environment: {e}")
        if py_eval_env is not None and hasattr(py_eval_env, 'close'):
            py_eval_env.close()
        return None, None, None, None, None, None, None

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

    actual_num_drones_in_env = eval_env.batch_size

    trajectories_unscaled_all_drones = [[] for _ in range(actual_num_drones_in_env)]
    target_trajectories_all_drones   = [[] for _ in range(actual_num_drones_in_env)]
    actions_raw_all_drones           = [[] for _ in range(actual_num_drones_in_env)]
    obs_shared_scaled_all_drones     = [[] for _ in range(actual_num_drones_in_env)]
    obs_task_scaled_all_drones       = [[] for _ in range(actual_num_drones_in_env)]
    rewards_all_drones               = [[] for _ in range(actual_num_drones_in_env)]
    total_rewards_accumulated_all_drones = np.zeros(actual_num_drones_in_env, dtype=np.float32)

    active_drones = [True] * actual_num_drones_in_env
    time_step = eval_env.reset()
    initial_shared = time_step.observation['shared_obs'].numpy()
    print("[Visualizer] shared_obs[0,:3] =", initial_shared[0, :3])
    print("[Visualizer] unscaled pos    =", unscale_position(initial_shared[0, :3],
                                                            OBS_POS_MIN_NP, OBS_POS_MAX_NP))
    policy_state = loaded_policy.get_initial_state(eval_env.batch_size)

    if 'shared_obs' not in time_step.observation or \
            'task_specific_obs' not in time_step.observation:
        print("Error: Initial observation structure is incorrect. Missing 'shared_obs' or 'task_specific_obs'.")
        if hasattr(eval_env, 'close'):
            eval_env.close()
        return None, None, None, None, None, None, None

    initial_shared_obs_batch = time_step.observation['shared_obs'].numpy()
    initial_task_obs_batch   = time_step.observation['task_specific_obs'].numpy()

    for i in range(actual_num_drones_in_env):
        obs_shared_scaled_all_drones[i].append(initial_shared_obs_batch[i].copy())
        obs_task_scaled_all_drones[i].append(initial_task_obs_batch[i].copy())

        scaled_start_position = initial_shared_obs_batch[i, :3]
        unscaled_start_position = unscale_position(scaled_start_position,
                                                   OBS_POS_MIN_NP,
                                                   OBS_POS_MAX_NP)
        trajectories_unscaled_all_drones[i].append(unscaled_start_position.tolist())

        # p_rel at t0 -> target position (for info/debug; not used for analytic traj)
        p_rel0 = initial_task_obs_batch[i, 0:3]
        p_rel_max_abs = np.max(np.abs(p_rel0))
        print(f"[Visualizer] p_rel0 = {p_rel0}, max_abs = {p_rel_max_abs:.3f}, "
              f"interpreting as {'SCALED' if p_rel_max_abs <= 1.0 else 'UNSCALED'}.")

        target_pos0 = unscaled_start_position + p_rel0
        target_trajectories_all_drones[i].append(target_pos0.tolist())

    print(f"Running policy for up to {num_steps} steps for {actual_num_drones_in_env} drones...")
    final_step_num = 0
    try:
        for step_num in range(num_steps):
            final_step_num = step_num
            if not np.any(active_drones):
                print(f"All drones finished before {num_steps} steps. "
                      f"Stopping at step {step_num}.")
                break

            action_step = loaded_policy.action(time_step, policy_state)
            current_actions_batch = action_step.action.numpy()
            time_step = eval_env.step(action_step.action)

            rewards_batch         = time_step.reward.numpy()
            next_shared_obs_batch = time_step.observation['shared_obs'].numpy()
            next_task_obs_batch   = time_step.observation['task_specific_obs'].numpy()
            is_last_batch         = time_step.is_last().numpy()

            for i in range(actual_num_drones_in_env):
                if not active_drones[i]:
                    continue

                actions_raw_all_drones[i].append(current_actions_batch[i].copy())

                r_i = rewards_batch[i]
                rewards_all_drones[i].append(r_i)
                total_rewards_accumulated_all_drones[i] += r_i

                obs_shared_scaled_all_drones[i].append(next_shared_obs_batch[i].copy())
                obs_task_scaled_all_drones[i].append(next_task_obs_batch[i].copy())

                # Drone position in world coords
                scaled_drone_pos = next_shared_obs_batch[i, :3]
                drone_pos_world = unscale_position(
                    scaled_drone_pos,
                    OBS_POS_MIN_NP,
                    OBS_POS_MAX_NP
                )
                trajectories_unscaled_all_drones[i].append(drone_pos_world.tolist())

                # Target position from p_rel_world (for info/debug only)
                p_rel = next_task_obs_batch[i, 0:3]   # world displacement (m)
                target_pos_world = drone_pos_world + p_rel
                target_trajectories_all_drones[i].append(target_pos_world.tolist())

                if is_last_batch[i]:
                    active_drones[i] = False
                    print(f"Drone {i} finished at environment step {step_num + 1} "
                          f"(Recorded {len(actions_raw_all_drones[i])} actions).")
            policy_state = action_step.state
    finally:
        if hasattr(eval_env, 'close'):
            eval_env.close()
            print("TFPyEnvironment closed.")
        elif py_eval_env is not None and hasattr(py_eval_env, 'close'):
            py_eval_env.close()
            print("PyEnvironment closed.")

    print(f"\nFinished policy run after {final_step_num + 1} potential environment steps.")
    print("Collected data summary:")
    for i in range(actual_num_drones_in_env):
        print(f"  Drone {i}: Ran for {len(actions_raw_all_drones[i])} action steps. "
              f"Total Reward: {total_rewards_accumulated_all_drones[i]:.4f}. "
              f"Drone pts: {len(trajectories_unscaled_all_drones[i])}, "
              f"Target pts: {len(target_trajectories_all_drones[i])}")

    # Convert lists-of-lists to arrays per drone
    actions_final = [
        np.array(actions) if len(actions) > 0 else np.empty((0,))
        for actions in actions_raw_all_drones
    ]
    obs_shared_final = [
        np.array(obs) if len(obs) > 0 else np.empty((0,))
        for obs in obs_shared_scaled_all_drones
    ]
    obs_task_final = [
        np.array(obs) if len(obs) > 0 else np.empty((0,))
        for obs in obs_task_scaled_all_drones
    ]
    rewards_final = [
        np.array(rewards) if len(rewards) > 0 else np.empty((0,))
        for rewards in rewards_all_drones
    ]

    eval_env.reset()

    return (trajectories_unscaled_all_drones,
            actions_final,
            obs_shared_final,
            obs_task_final,
            rewards_final,
            total_rewards_accumulated_all_drones,
            target_trajectories_all_drones)


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
                           target_trajectories_list=None,
                           title="3D Interception Scenario",
                           normal_vector_length=0.5):
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

        # --- Show intended target direction (matches C++ config if you align TARGET_VEL_WORLD) ---
        # Here we visualize TARGET_VEL_WORLD as the arrow from gate center
        direction = TARGET_VEL_WORLD.astype(float)
        norm_d = np.linalg.norm(direction)
        if norm_d > 1e-6:
            direction = direction / norm_d
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)

        ax.quiver(center_point[0], center_point[1], center_point[2],
                  direction[0], direction[1], direction[2],
                  length=normal_vector_length, color='darkmagenta',
                  arrow_length_ratio=0.3,
                  label='Target Vel Dir' if i == 0 else "")

        all_x.extend(gate_corners_np[:, 0])
        all_y.extend(gate_corners_np[:, 1])
        all_z.extend(gate_corners_np[:, 2])

    # --- Plot drone trajectories ---
    if trajectories_list:
        num_trajectories = len(trajectories_list)
        colors_map = plt.cm.get_cmap('viridis', num_trajectories if num_trajectories > 0 else 1)

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

    # --- Plot target trajectories (analytic from gate + TARGET_VEL_WORLD) ---
    if target_trajectories_list:
        num_target_trajs = len(target_trajectories_list)
        target_cmap = plt.cm.get_cmap('plasma', num_target_trajs if num_target_trajs > 0 else 1)

        for idx, target_traj_points in enumerate(target_trajectories_list):
            if target_traj_points and len(target_traj_points) > 1:
                targ_np = np.array(target_traj_points)
                num_steps_for_traj = len(targ_np) - 1
                targ_color = target_cmap(idx / num_target_trajs) if num_target_trajs > 1 else 'red'
                ax.plot(targ_np[:, 0], targ_np[:, 1], targ_np[:, 2],
                        color=targ_color, linewidth=1,
                        label=f'Target {idx} Traj ({num_steps_for_traj} steps)',
                        linestyle='--',
                        marker='.', markersize=1)
                all_x.extend(targ_np[:, 0])
                all_y.extend(targ_np[:, 1])
                all_z.extend(targ_np[:, 2])
                ax.scatter(targ_np[0, 0], targ_np[0, 1], targ_np[0, 2],
                           color=targ_color, s=60, marker='s',
                           label=f'T{idx} Start', depthshade=False, zorder=9)
                ax.scatter(targ_np[-1, 0], targ_np[-1, 1], targ_np[-1, 2],
                           color=targ_color, s=60, marker='D',
                           label=f'T{idx} End', depthshade=False, zorder=9)

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

    # Dedup legend
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
            if "." in os.path.basename(p_val):
                if not os.path.exists(p_val):
                    print(f"Creating dummy file: {p_val}")
                    open(p_val, 'w').close()
            else:
                if not os.path.exists(p_val):
                    print(f"Creating dummy directory: {p_val}")
                    os.makedirs(p_val, exist_ok=True)

    (all_drone_trajectories,
     all_raw_actions,
     all_scaled_shared_obs,
     all_scaled_task_obs,
     all_rewards_data,
     all_total_rewards,
     all_target_trajectories) = (None, None, None, None, None, None, None)

    policy_run_attempted = False
    if not os.path.exists(POLICY_DIR) or not os.listdir(POLICY_DIR):
        print(f"Warning: Policy directory '{POLICY_DIR}' is empty or does not exist. Skipping policy run.")
    elif BatchedAgiSimEnv is None:
        print(f"Warning: BatchedAgiSimEnv not available. Cannot run policy.")
    else:
        policy_run_attempted = True
        print(f"Attempting to run policy for {NUM_DRONES_TO_VISUALIZE} drone(s).")
        (all_drone_trajectories,
         all_raw_actions,
         all_scaled_shared_obs,
         all_scaled_task_obs,
         all_rewards_data,
         all_total_rewards,
         all_target_trajectories) = run_policy_and_collect_trajectory(
            POLICY_DIR, env_params_for_run, NUM_EVAL_STEPS, NUM_DRONES_TO_VISUALIZE
        )
    # Reconstruct target trajectories from gate center and constant velocity (must match C++ config)
    if policy_run_attempted and all_drone_trajectories is not None:
        all_target_trajectories = []
        # Gate center from first gate in track.json
        if final_track_layout and len(final_track_layout[0]) > 0:
            gate0 = np.array(final_track_layout[0], dtype=np.float32)
            gate_center = gate0.mean(axis=0)
        else:
            gate_center = np.array([150.0, 0.5, 100.0], dtype=np.float32)

        for drone_traj in all_drone_trajectories:
            if not drone_traj:
                all_target_trajectories.append([])
                continue
            num_pts = len(drone_traj)
            timesteps = np.arange(num_pts, dtype=np.float32) * TARGET_DT
            target_positions = gate_center + np.outer(timesteps, TARGET_VEL_WORLD)
            all_target_trajectories.append(target_positions.tolist())


    # Debug distances between drone and target 0
    # Debug distances and truncate trajectories at capture (for plotting)
    if (policy_run_attempted and
        all_drone_trajectories is not None and
        all_target_trajectories is not None and
        len(all_drone_trajectories) > 0 and
        len(all_target_trajectories) > 0):

        print("\n=== Interception debug / capture truncation ===")
        for i, (drone_traj, target_traj) in enumerate(
                zip(all_drone_trajectories, all_target_trajectories)):

            if not drone_traj or not target_traj:
                print(f"Drone {i}: empty trajectory, skipping.")
                continue

            d_np = np.array(drone_traj, dtype=np.float32)
            t_np = np.array(target_traj, dtype=np.float32)

            min_len = min(d_np.shape[0], t_np.shape[0])
            d_np = d_np[:min_len]
            t_np = t_np[:min_len]

            dists = np.linalg.norm(t_np - d_np, axis=1)

            print(f"\nDrone {i}:")
            print(f"  num steps (raw): {min_len}")
            print(f"  min dist:  {dists.min():.3f} m")
            print(f"  max dist:  {dists.max():.3f} m")
            print(f"  mean dist: {dists.mean():.3f} m")

            # find first capture step
            hits = np.where(dists <= CAPTURE_RADIUS_M)[0]
            if hits.size > 0:
                cap_idx = int(hits[0])
                # ensure at least 2 points so the plotter actually draws a line
                trim_idx = max(cap_idx, 1)
                all_drone_trajectories[i] = d_np[:trim_idx + 1].tolist()
                all_target_trajectories[i] = t_np[:trim_idx + 1].tolist()
                print(f"  -> capture at step {cap_idx} "
                      f"(dist={dists[cap_idx]:.3f} m <= {CAPTURE_RADIUS_M} m); "
                      f"truncating to {trim_idx + 1} points for plotting.")
            else:
                # no capture; keep full trajectory
                all_drone_trajectories[i] = d_np.tolist()
                all_target_trajectories[i] = t_np.tolist()
                print(f"  -> no capture within {CAPTURE_RADIUS_M} m; "
                      "keeping full trajectory.")

            # print a few sample points for sanity
            for k in [0, 1, min_len // 2, min_len - 2, min_len - 1]:
                if 0 <= k < min_len:
                    print(f"    t={k:3d}: "
                          f"d={d_np[k]}, "
                          f"t={t_np[k]}, "
                          f"dist={dists[k]:.3f}")
        print("===============================================")

    # Visualize the 3D track and all drone trajectories
    visualize_track_layout(
        final_track_layout,
        trajectories_list=all_drone_trajectories,
        target_trajectories_list=all_target_trajectories,
        title=f"{NUM_DRONES_TO_VISUALIZE if policy_run_attempted and all_drone_trajectories else 0} "
              f"Drone(s) Interception Trajectories"
    )

    # --- Detailed 2D plots for the best drone ---
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

    elif policy_run_attempted:
        print("\n--- Skipping 2D plots as data from policy run is incomplete, inconsistent, or empty. ---")
    else:
        print("\n--- Skipping 2D plots as policy run was not performed. ---")

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
        print("  No drone met criteria for detailed 2D plotting.")

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

    if plt.get_fignums():
        print(f"\nDisplaying {len(plt.get_fignums())} plot window(s)...")
        plt.show()
    else:
        print("\nNo plots were generated to display.")


if __name__ == "__main__":
    # Basic check to ensure the script can run even if paths are not fully set up
    if not os.path.exists(POLICY_DIR):
        print(f"INFO: POLICY_DIR '{POLICY_DIR}' does not exist. Creating for test run.")
        os.makedirs(POLICY_DIR, exist_ok=True)
        if not os.path.exists(os.path.join(POLICY_DIR, "saved_model.pb")):
            open(os.path.join(POLICY_DIR, "saved_model.pb"), 'w').close()
    main()
