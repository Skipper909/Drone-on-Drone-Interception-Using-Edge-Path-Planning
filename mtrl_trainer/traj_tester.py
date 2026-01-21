import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For 3D polygons
import os

# --- TF-Agents and Environment Imports ---
# Ensure these libraries are installed: pip install tf-agents
# Ensure your custom environment files are accessible.
try:
    from mtrl_lib.agisim_environment import BatchedAgiSimEnv
    # from auto_reset_wrapper import AutoResetWrapper # Not strictly needed for this test
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv.")
    print("Please ensure agisim_environment.py is in the Python path or the same directory.")
    BatchedAgiSimEnv = None

# --- Configuration ---
# These should match the settings your C++ environment expects/was compiled with
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters')

# Define the track layout. This is important for testing gate-related rewards.
TRACK_LAYOUT_TEST = [
    #[[1.0, 1.0, 3.0], [-1.0, 1.0, 3.0], [-1.0, -1.0, 3.0], [1.0, -1.0, 3.0]],
    #[[2.9226, 1.0000, 4.3937], [2.0774, 1.0000, 6.2063], [2.0774, -1.0000, 6.2063], [2.9226, -1.0000, 4.3937]],
    #[[5.0000, 1.0000, 5.0000], [5.0000, 1.0000, 7.0000], [5.0000, -1.0000, 7.0000], [5.0000, -1.0000, 5.0000]],
    #[[7.0000, 5.0000, 5.0000], [7.0000, 5.0000, 7.0000], [9.0000, 5.0000, 7.0000], [9.0000, 5.0000, 5.0000]],
    #[[5.0000, 9.0000, 5.0000], [5.0000, 9.0000, 7.0000], [5.0000, 11.0000, 7.0000], [5.0000, 11.0000, 5.0000]],
    #[[-3.0000, 16.0000, 7.0000], [-3.0000, 16.0000, 9.0000], [-1.0000, 16.0000, 9.0000], [-1.0000, 16.0000, 7.0000]],
    #[[-5.0000, 21.0000, 4.0000], [-3.0000, 21.0000, 4.0000], [-3.0000, 19.0000, 4.0000], [-5.0000, 19.0000, 4.0000]]

    [[7.0, -2.0, 1.9999999999999996], [7.0, -1.9999999999999996, 4.0], [7.0, -4.0, 4.0], [7.0, -4.0, 2.0]], [[10.0, -0.000199999999999889, 2.0], [10.0, -0.0002, 4.0], [12.0, -0.00020000000000011103, 4.0], [12.0, -0.0002, 2.0]], [[7.0, 2.0, 3.0], [7.0, 2.0, 5.0], [7.0, 4.0, 5.0], [7.0, 4.0, 3.0]], [[-1.9824991799798666, -3.999846848921685, 3.0], [-1.9824991799798664, -3.999846848921685, 5.0], [-2.0175008200201336, -2.000153151078315, 5.0], [-2.0175008200201336, -2.000153151078315, 3.0]], [[-6.99938913597975, 0.03494788814289562, 2.0], [-6.99938913597975, 0.034947888142895844, 4.0], [-5.00061086402025, -0.03494788814289562, 4.0], [-5.00061086402025, -0.034947888142895844, 2.0]], [[-2.0, 4.0, 1.9999999999999996], [-2.0, 4.0, 4.0], [-2.0, 2.0, 4.0], [-2.0, 1.9999999999999996, 2.0]]


]


NUM_TEST_STEPS = 800 # Number of steps to run the hardcoded actions
NORMAL_VECTOR_LENGTH_3D_PLOT = 1 # User specified

# --- IMPORTANT: Define Position Scaling Parameters ---
# These MUST match the OBS_POS_MIN and OBS_POS_MAX used in your C++ RewardParams
# for scaling the position observation.
OBS_POS_MIN_NP = np.array([-120.0, -120.0, -120.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([120.0, 120.0, 120.0], dtype=np.float32)


def define_action_sequence(num_steps):
    """
    Defines a sequence of actions.
    Action: [thrust, roll_rate, pitch_rate, yaw_rate] (all in [-1, 1] for policy, except thrust [0,1])
    """
    actions = []
    # Scenario 1: Fly towards Gate 0, try to pass
    #for i in range(200): # Approach Gate 0
    #    actions.append(np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float32))
    #for i in range(500): # Approach Gate 0
    #    actions.append(np.array([0.5, 0.266, 0.0, 0.0], dtype=np.float32))
    # Scenario 2: Attempt a turn towards Gate 1 (e.g., left yaw and some roll)
    #for i in range(300): # Initiate turn
    #    actions.append(np.array([1.0, 0.65, 0.0, 0.0], dtype=np.float32)) # Yaw left, some roll left


    for i in range(500): # Approach Gate 0
        actions.append(np.array([0.0, 0.0, 0, 2.0], dtype=np.float32))

   # for i in range(500): # Approach Gate 0
    #    actions.append(np.array([0.7, 0.18, -0.18, 0.0], dtype=np.float32))

    num_landing_steps = num_steps - len(actions)
    if num_landing_steps < 0: num_landing_steps = 0
    for i in range(num_landing_steps):
        actions.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return actions[:num_steps]


def unscale_pos(scaled_pos_np, obs_pos_min_param, obs_pos_max_param):
    """ Unscales a 3D position vector. """
    range_bounds = obs_pos_max_param - obs_pos_min_param
    for i in range(len(range_bounds)): # Avoid division by zero
        if np.isclose(range_bounds[i], 0.0): range_bounds[i] = 1.0 # Should not happen with valid ranges
    unscaled_pos = ((scaled_pos_np + 1.0) * range_bounds / 2.0) + obs_pos_min_param
    return unscaled_pos

def run_reward_test():
    if BatchedAgiSimEnv is None:
        print("Aborting: BatchedAgiSimEnv not imported.")
        return

    print("Initializing BatchedAgiSimEnv for reward testing (num_drones=1)...")
    try:
        env = BatchedAgiSimEnv(
            sim_config_path=SIM_CONFIG_PATH,
            agi_param_dir=AGI_PARAM_DIR,
            sim_base_dir=SIM_BASE_DIR,
            num_drones=1,
            track_layout=TRACK_LAYOUT_TEST
        )
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    action_sequence = define_action_sequence(NUM_TEST_STEPS)

    rewards_over_time = []
    drone_z_positions = []
    drone_x_positions = []
    drone_y_positions = []
    target_gate_indices_log = []
    all_observations_log = [] # To store full scaled observations
    total_trajectory_reward = 0.0

    print("Resetting environment...")
    time_step = env.reset() # time_step.observation is now a dict

    current_obs_pos_min = OBS_POS_MIN_NP
    current_obs_pos_max = OBS_POS_MAX_NP

    # --- MODIFIED: Correctly extract and combine observation for the single drone ---
    if not isinstance(time_step.observation, dict) or 'shared_obs' not in time_step.observation or 'task_specific_obs' not in time_step.observation:
        print("Error: time_step.observation is not the expected dictionary structure.")
        print(f"Observation content: {time_step.observation}")
        return

    shared_obs_drone0 = time_step.observation['shared_obs'][0] # Shape (19,)
    task_specific_obs_drone0 = time_step.observation['task_specific_obs'][0] # Shape (24,)
    initial_observation_scaled = np.concatenate((shared_obs_drone0, task_specific_obs_drone0)) # Shape (43,)
    # --- END MODIFICATION ---

    all_observations_log.append(initial_observation_scaled.copy()) # Store a copy
    initial_pos_scaled = initial_observation_scaled[:3] # This will now work correctly
    initial_pos_unscaled = unscale_pos(initial_pos_scaled, current_obs_pos_min, current_obs_pos_max)

    drone_x_positions.append(initial_pos_unscaled[0])
    drone_y_positions.append(initial_pos_unscaled[1])
    drone_z_positions.append(initial_pos_unscaled[2])
    try:
        target_gate_indices_log.append(env._env.target_gate_indices_[0])
        print(f"Initial drone_pos (unscaled): {initial_pos_unscaled.tolist()}")
        print(f"Initial target_gate_idx: {target_gate_indices_log[-1]}")
    except AttributeError:
        print(f"Could not access target_gate_indices_ from C++ env directly.")
        target_gate_indices_log.append(0) # Default or placeholder
    except Exception as e:
        print(f"Error accessing target_gate_indices_ from C++ env: {e}")
        target_gate_indices_log.append(0)


    for step_num, action in enumerate(action_sequence):
        time_step = env.step(np.expand_dims(action, axis=0)) # time_step.observation is a dict

        reward = time_step.reward[0] # Assuming reward is still a flat array[0]
        rewards_over_time.append(float(reward))
        total_trajectory_reward += float(reward)

        # --- MODIFIED: Correctly extract and combine observation for the single drone ---
        if not isinstance(time_step.observation, dict) or 'shared_obs' not in time_step.observation or 'task_specific_obs' not in time_step.observation:
            print("Error: Subsequent time_step.observation is not the expected dictionary structure.")
            break

        shared_obs_drone0_current = time_step.observation['shared_obs'][0]
        task_specific_obs_drone0_current = time_step.observation['task_specific_obs'][0]
        current_observation_scaled = np.concatenate((shared_obs_drone0_current, task_specific_obs_drone0_current))
        # --- END MODIFICATION ---

        all_observations_log.append(current_observation_scaled.copy()) # Store a copy
        current_pos_scaled = current_observation_scaled[:3] # This will now work
        current_pos_unscaled = unscale_pos(current_pos_scaled, current_obs_pos_min, current_obs_pos_max)

        drone_x_positions.append(current_pos_unscaled[0])
        drone_y_positions.append(current_pos_unscaled[1])
        drone_z_positions.append(current_pos_unscaled[2])

        try:
            current_target_idx = env._env.target_gate_indices_[0]
        except AttributeError:
            current_target_idx = target_gate_indices_log[-1] if target_gate_indices_log else 0
        except Exception:
            current_target_idx = target_gate_indices_log[-1] if target_gate_indices_log else 0
        target_gate_indices_log.append(current_target_idx)

        if step_num % 50 == 0:

            obs_str_parts = [
                f"  Obs (Scaled): P={current_observation_scaled[0:3].tolist()}",
                f"R_col0={current_observation_scaled[3:6].tolist()}",
                f"R_col1={current_observation_scaled[6:9].tolist()}",
                f"V={current_observation_scaled[9:12].tolist()}",
                f"AngV={current_observation_scaled[12:15].tolist()}",
                f"PrevAct={current_observation_scaled[15:19].tolist()}",
                f"dltP1_c0={current_observation_scaled[19:22].tolist()}",
                f"dltP2_c0={current_observation_scaled[31:34].tolist()}"
            ]

        if time_step.is_last()[0]: # Assuming batch_size=1, so is_last() is an array like [True]
            print(f"Episode ended at step {step_num + 1}.")
            remaining_steps = NUM_TEST_STEPS - (step_num + 1)
            if remaining_steps > 0: # Check if there are remaining steps to fill
                rewards_over_time.extend([float(reward)] * remaining_steps)
                drone_x_positions.extend([current_pos_unscaled[0]] * remaining_steps)
                drone_y_positions.extend([current_pos_unscaled[1]] * remaining_steps)
                drone_z_positions.extend([current_pos_unscaled[2]] * remaining_steps)
                target_gate_indices_log.extend([current_target_idx] * remaining_steps)
                all_observations_log.extend([current_observation_scaled.copy()] * remaining_steps) # Store copies
            break
    if 'env' in locals() and hasattr(env, 'close'): # Ensure env was initialized
        env.close()

    all_observations_np = np.array(all_observations_log)

    # --- Plotting ---
    # 2D Plots (Rewards, Z-position, Target Gate)
    fig_2d, axs_2d = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs_2d[0].plot(rewards_over_time, label='Reward per Step', color='blue', marker='.')
    axs_2d[0].set_ylabel('Reward'); axs_2d[0].set_title('Reward Function Test (2D Info)'); axs_2d[0].grid(True); axs_2d[0].legend()
    axs_2d[1].plot(drone_z_positions, label='Drone Z Position (World)', color='green', marker='.')
    axs_2d[1].set_ylabel('Z Position (m)'); axs_2d[1].grid(True); axs_2d[1].legend()
    axs_2d[2].plot(target_gate_indices_log, label='Target Gate Index', color='red', linestyle='--', marker='o')
    axs_2d[2].set_ylabel('Target Gate Index'); axs_2d[2].set_xlabel('Time Step'); axs_2d[2].grid(True); axs_2d[2].legend()
    plt.tight_layout()

    # 3D Plot (Track, Normals, and Trajectory)
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    plot_all_x, plot_all_y, plot_all_z = [], [], []
    for i, gate_corners_list in enumerate(TRACK_LAYOUT_TEST):
        gate_corners_np = np.array(gate_corners_list)
        gate_polygon = Poly3DCollection([gate_corners_np], alpha=0.3, linewidths=1, edgecolors='k')
        plot_colors = ['r', 'g', 'b', 'y', 'c', 'm']
        gate_polygon.set_facecolor(plot_colors[i % len(plot_colors)])
        ax_3d.add_collection3d(gate_polygon)
        center_point = np.mean(gate_corners_np, axis=0)
        ax_3d.text(center_point[0], center_point[1], center_point[2], f" G{i}", color='k', fontsize=8)
        v1 = gate_corners_np[1] - gate_corners_np[0]
        v2 = gate_corners_np[3] - gate_corners_np[0]
        normal_vector = np.cross(v1, v2)
        norm_magnitude = np.linalg.norm(normal_vector)
        if norm_magnitude > 1e-6: normal_vector = normal_vector / norm_magnitude
        else: normal_vector = np.array([0,0,1])
        ax_3d.quiver( center_point[0], center_point[1], center_point[2], normal_vector[0], normal_vector[1], normal_vector[2], length=NORMAL_VECTOR_LENGTH_3D_PLOT, color='purple', arrow_length_ratio=0.3, label='Gate Normal' if i == 0 else "")
        plot_all_x.extend(gate_corners_np[:, 0]); plot_all_y.extend(gate_corners_np[:, 1]); plot_all_z.extend(gate_corners_np[:, 2])
    if drone_x_positions and len(drone_x_positions) > 1:
        ax_3d.plot(drone_x_positions, drone_y_positions, drone_z_positions, color='blue', linewidth=2, label='Drone Trajectory', marker='.', markersize=3)
        plot_all_x.extend(drone_x_positions); plot_all_y.extend(drone_y_positions); plot_all_z.extend(drone_z_positions)
        ax_3d.scatter(drone_x_positions[0], drone_y_positions[0], drone_z_positions[0], color='lime', s=100, marker='o', label=f'Traj Start ({drone_x_positions[0]:.1f},{drone_y_positions[0]:.1f},{drone_z_positions[0]:.1f})', depthshade=False, zorder=10)
        ax_3d.scatter(drone_x_positions[-1], drone_y_positions[-1], drone_z_positions[-1], color='orange', s=100, marker='X', label='Traj End', depthshade=False, zorder=10)
    dot_x, dot_y, dot_z = 0, 0, 1
    ax_3d.scatter([dot_x], [dot_y], [dot_z], color='black', s=50, marker='P', label='Reference Point (0,0,1)')
    if not plot_all_x: plot_all_x.extend([dot_x-1, dot_x+1]); plot_all_y.extend([dot_y-1, dot_y+1]); plot_all_z.extend([dot_z-1, dot_z+1])
    else: plot_all_x.append(dot_x); plot_all_y.append(dot_y); plot_all_z.append(dot_z)
    if not plot_all_x: ax_3d.set_xlim([-1, 1]); ax_3d.set_ylim([-1, 1]); ax_3d.set_zlim([0, 2])
    else:
        ax_3d.set_xlim(min(plot_all_x) - 1, max(plot_all_x) + 1)
        ax_3d.set_ylim(min(plot_all_y) - 1, max(plot_all_y) + 1)
        ax_3d.set_zlim(min(plot_all_z) - 1, max(plot_all_z) + 1)
    ax_3d.set_xlabel('X-axis'); ax_3d.set_ylabel('Y-axis'); ax_3d.set_zlabel('Z-axis')
    ax_3d.set_title('3D Drone Trajectory and Track Layout')
    ax_3d.legend()
    x_limits_3d = ax_3d.get_xlim3d(); y_limits_3d = ax_3d.get_ylim3d(); z_limits_3d = ax_3d.get_zlim3d()
    x_range_3d = abs(x_limits_3d[1] - x_limits_3d[0]); x_middle_3d = np.mean(x_limits_3d)
    y_range_3d = abs(y_limits_3d[1] - y_limits_3d[0]); y_middle_3d = np.mean(y_limits_3d)
    z_range_3d = abs(z_limits_3d[1] - z_limits_3d[0]); z_middle_3d = np.mean(z_limits_3d)
    plot_radius_3d = 0.5 * max([x_range_3d, y_range_3d, z_range_3d])
    ax_3d.set_xlim3d([x_middle_3d - plot_radius_3d, x_middle_3d + plot_radius_3d])
    ax_3d.set_ylim3d([y_middle_3d - plot_radius_3d, y_middle_3d + plot_radius_3d])
    ax_3d.set_zlim3d([z_middle_3d - plot_radius_3d, z_middle_3d + plot_radius_3d])

    # --- Plotting Full Observation Vector ---
    if all_observations_np.shape[1] == 43: # Check if we have 43 components
        fig_obs, axs_obs = plt.subplots(8, 1, figsize=(15, 20), sharex=True)
        time_steps_plot = np.arange(all_observations_np.shape[0])

        # 1. Position (scaled)
        axs_obs[0].plot(time_steps_plot, all_observations_np[:, 0], label='Pos X (scaled)')
        axs_obs[0].plot(time_steps_plot, all_observations_np[:, 1], label='Pos Y (scaled)')
        axs_obs[0].plot(time_steps_plot, all_observations_np[:, 2], label='Pos Z (scaled)')
        axs_obs[0].set_ylabel('Position (scaled)')
        axs_obs[0].set_title('Scaled Observation Components Over Time')
        axs_obs[0].legend(); axs_obs[0].grid(True)
        axs_obs[0].set_ylim(-1.05, 1.05) # MODIFIED

        # 2. R_wb col0 (scaled, though typically already [-1,1])
        axs_obs[1].plot(time_steps_plot, all_observations_np[:, 3], label='R_wb[0,0]')
        axs_obs[1].plot(time_steps_plot, all_observations_np[:, 4], label='R_wb[1,0]')
        axs_obs[1].plot(time_steps_plot, all_observations_np[:, 5], label='R_wb[2,0]')
        axs_obs[1].set_ylabel('R_wb col0')
        axs_obs[1].legend(); axs_obs[1].grid(True)
        axs_obs[1].set_ylim(-1.55, 1.55) # MODIFIED

        # 3. R_wb col1 (scaled, though typically already [-1,1])
        axs_obs[2].plot(time_steps_plot, all_observations_np[:, 6], label='R_wb[0,1]')
        axs_obs[2].plot(time_steps_plot, all_observations_np[:, 7], label='R_wb[1,1]')
        axs_obs[2].plot(time_steps_plot, all_observations_np[:, 8], label='R_wb[2,1]')
        axs_obs[2].set_ylabel('R_wb col1')
        axs_obs[2].legend(); axs_obs[2].grid(True)
        axs_obs[2].set_ylim(-1.55, 1.55) # MODIFIED

        # 4. Velocity (scaled)
        axs_obs[3].plot(time_steps_plot, all_observations_np[:, 9], label='Vel X (scaled)')
        axs_obs[3].plot(time_steps_plot, all_observations_np[:, 10], label='Vel Y (scaled)')
        axs_obs[3].plot(time_steps_plot, all_observations_np[:, 11], label='Vel Z (scaled)')
        axs_obs[3].set_ylabel('Velocity (scaled)')
        axs_obs[3].legend(); axs_obs[3].grid(True)
        axs_obs[3].set_ylim(-1.05, 1.05) # MODIFIED

        # 5. Angular Velocity (scaled)
        axs_obs[4].plot(time_steps_plot, all_observations_np[:, 12], label='AngVel X (scaled)')
        axs_obs[4].plot(time_steps_plot, all_observations_np[:, 13], label='AngVel Y (scaled)')
        axs_obs[4].plot(time_steps_plot, all_observations_np[:, 14], label='AngVel Z (scaled)')
        axs_obs[4].set_ylabel('AngVel (scaled)')
        axs_obs[4].legend(); axs_obs[4].grid(True)
        axs_obs[4].set_ylim(-1.05, 1.05) # MODIFIED

        # 6. Previous Action (raw, typically [-1,1] or [0,1])
        axs_obs[5].plot(time_steps_plot, all_observations_np[:, 15], label='PrevAct Thrust')
        axs_obs[5].plot(time_steps_plot, all_observations_np[:, 16], label='PrevAct RateX')
        axs_obs[5].plot(time_steps_plot, all_observations_np[:, 17], label='PrevAct RateY')
        axs_obs[5].plot(time_steps_plot, all_observations_np[:, 18], label='PrevAct RateZ')
        axs_obs[5].set_ylabel('Previous Action')
        axs_obs[5].legend(); axs_obs[5].grid(True)
        axs_obs[5].set_ylim(-1.05, 1.05) # MODIFIED

        # 7. Delta_p1 norms (scaled body-frame relative corners)
        dp1_norms = np.zeros((all_observations_np.shape[0], 4))
        for t_idx in range(all_observations_np.shape[0]):
            for c_idx in range(4):
                start = 19 + c_idx * 3
                dp1_norms[t_idx, c_idx] = np.linalg.norm(all_observations_np[t_idx, start:start+3])
        axs_obs[6].plot(time_steps_plot, dp1_norms[:, 0], label='||dP1_c0|| (body)')
        axs_obs[6].plot(time_steps_plot, dp1_norms[:, 1], label='||dP1_c1|| (body)')
        axs_obs[6].plot(time_steps_plot, dp1_norms[:, 2], label='||dP1_c2|| (body)')
        axs_obs[6].plot(time_steps_plot, dp1_norms[:, 3], label='||dP1_c3|| (body)')
        axs_obs[6].set_ylabel('Delta_p1 Norms (scaled)')
        axs_obs[6].legend(); axs_obs[6].grid(True)
        axs_obs[6].set_ylim(-1.05, 1.05) # MODIFIED

        # 8. Delta_p2 norms (scaled world-frame gate-to-gate relative corners)
        dp2_norms = np.zeros((all_observations_np.shape[0], 4))
        for t_idx in range(all_observations_np.shape[0]):
            for c_idx in range(4):
                start = 31 + c_idx * 3
                dp2_norms[t_idx, c_idx] = np.linalg.norm(all_observations_np[t_idx, start:start+3])
        axs_obs[7].plot(time_steps_plot, dp2_norms[:, 0], label='||dP2_c0|| (world)')
        axs_obs[7].plot(time_steps_plot, dp2_norms[:, 1], label='||dP2_c1|| (world)')
        axs_obs[7].plot(time_steps_plot, dp2_norms[:, 2], label='||dP2_c2|| (world)')
        axs_obs[7].plot(time_steps_plot, dp2_norms[:, 3], label='||dP2_c3|| (world)')
        axs_obs[7].set_ylabel('Delta_p2 Norms (scaled)')
        axs_obs[7].set_xlabel('Time Step')
        axs_obs[7].legend(); axs_obs[7].grid(True)
        axs_obs[7].set_ylim(-1.05, 1.05) # MODIFIED

        plt.tight_layout()
    else:
        print(f"Warning: Observation data not in expected shape (43 components) for detailed plotting. Shape: {all_observations_np.shape}")


    plt.show()


if __name__ == '__main__':
    run_reward_test()
