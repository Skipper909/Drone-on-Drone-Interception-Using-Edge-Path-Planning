import os
import numpy as np
import tensorflow as tf # Still needed for TFPyEnvironment if used
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Attempt to import your custom environment
try:
    from mtrl_trainer.mtrl_lib.agisim_environment import BatchedAgiSimEnv
    # AutoResetWrapper is not strictly needed for a single reset visualization
except ImportError:
    print("WARNING: Could not import BatchedAgiSimEnv.")
    print("Please ensure these files are in the Python path or the same directory.")
    BatchedAgiSimEnv = None

# --- Configuration ---
# Update these paths and track layout as needed
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'simulator', 'parameters')

# Example TRACK_LAYOUT (replace with your actual track data or load it)
TRACK_LAYOUT = [
    [[7.0000, -2.0000, 2.0000], [7.0000, -2.0000, 4.0000], [7.0000, -4.0000, 4.0000], [7.0000, -4.0000, 2.0000]],
    [[10.0000, 0.0000, 2.0000], [10.0000, 0.0000, 4.0000], [12.0000, -0.0000, 4.0000], [12.0000, 0.0000, 2.0000]],
    [[7.0000, 2.0000, 3.0000], [7.0000, 2.0000, 5.0000], [7.0000, 4.0000, 5.0000], [7.0000, 4.0000, 3.0000]],
    [[-1.9825, -3.9998, 3.0000], [-1.9825, -3.9998, 5.0000], [-2.0175, -2.0002, 5.0000], [-2.0175, -2.0002, 3.0000]],
    [[-7.0000, 0.0000, 2.0000], [-7.0000, 0.0000, 4.0000], [-5.0000, -0.0000, 4.0000], [-5.0000, 0.0000, 2.0000]],
    [[-2.0000, 4.0000, 2.0000], [-2.0000, 4.0000, 4.0000], [-2.0000, 2.0000, 4.0000], [-2.0000, 2.0000, 2.0000]]
    # Add more gates if you have them
]

# Observation scaling parameters (should match your environment's C++ settings)
OBS_POS_MIN_NP = np.array([-30.0, -30.0, -30.0], dtype=np.float32)
OBS_POS_MAX_NP = np.array([30.0, 30.0, 30.0], dtype=np.float32)

def unscale_position(scaled_pos_np, min_bounds_np, max_bounds_np):
    """Unscales a 3D position vector from [-1, 1] to original world coordinates."""
    range_bounds = max_bounds_np - min_bounds_np
    for i in range(len(range_bounds)):
        if np.isclose(range_bounds[i], 0.0):
            range_bounds[i] = 1.0
    unscaled_pos = ((scaled_pos_np + 1.0) * range_bounds / 2.0) + min_bounds_np
    return unscaled_pos

def get_body_axes_from_shared_obs(shared_obs_numpy):
    """
    Extracts drone body axes (in world frame) from the shared observation.
    Observation contains the first two columns of R_WB (World-to-Body rotation matrix).
    """
    # R_WB transforms world vectors to body vectors. shared_obs[3:9] are first two cols of R_WB.
    # obs[3:6] is col0 of R_WB
    # obs[6:9] is col1 of R_WB
    col0_r_wb = shared_obs_numpy[3:6]
    col1_r_wb = shared_obs_numpy[6:9]

    # Calculate the third column of R_WB
    col2_r_wb = np.cross(col0_r_wb, col1_r_wb)

    # Form R_WB matrix (columns are col0_r_wb, col1_r_wb, col2_r_wb)
    r_wb = np.column_stack((col0_r_wb, col1_r_wb, col2_r_wb))

    # The body axes in world coordinates are the columns of R_BW = R_WB^T.
    # So, x_body_in_world is the first row of R_WB, etc.
    x_body_in_world = r_wb[0, :]
    y_body_in_world = r_wb[1, :]
    z_body_in_world = r_wb[2, :]

    return x_body_in_world, y_body_in_world, z_body_in_world

def visualize_drone_and_track(track_layout, drone_pos_world,
                              x_axis_world, y_axis_world, z_axis_world,
                              title="Drone State at Reset",
                              gate_normal_length=3.5, body_axis_length=3.3):
    """Visualizes gates, drone position, and drone orientation (body axes)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []

    # Plot Gates and their Normals
    for i, gate_corners_list in enumerate(track_layout):
        gate_corners_np = np.array(gate_corners_list)
        gate_polygon = Poly3DCollection([gate_corners_np], alpha=0.25, linewidths=1, edgecolors='k')
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        gate_polygon.set_facecolor(colors[i % len(colors)])
        ax.add_collection3d(gate_polygon)
        center_point = np.mean(gate_corners_np, axis=0)
        ax.text(center_point[0], center_point[1], center_point[2], f" G{i}", color='k', fontsize=8)

        v1 = gate_corners_np[1] - gate_corners_np[0]
        v2 = gate_corners_np[3] - gate_corners_np[0]
        normal_vector = np.cross(v1, v2)
        norm_magnitude = np.linalg.norm(normal_vector)
        if norm_magnitude > 1e-6:
            normal_vector /= norm_magnitude
        else:
            normal_vector = np.array([0,0,1]) # Default normal

        ax.quiver(center_point[0], center_point[1], center_point[2],
                  normal_vector[0], normal_vector[1], normal_vector[2],
                  length=gate_normal_length, color='darkslategray', arrow_length_ratio=0.3,
                  label='Gate Normal' if i == 0 else "")

        all_x.extend(gate_corners_np[:, 0]); all_y.extend(gate_corners_np[:, 1]); all_z.extend(gate_corners_np[:, 2])

    # Plot Drone Position
    ax.scatter(drone_pos_world[0], drone_pos_world[1], drone_pos_world[2],
               color='black', s=100, marker='o', label='Drone Position', depthshade=False, zorder=10)
    all_x.append(drone_pos_world[0]); all_y.append(drone_pos_world[1]); all_z.append(drone_pos_world[2])

    # Plot Drone Body Axes
    ax.quiver(drone_pos_world[0], drone_pos_world[1], drone_pos_world[2],
              x_axis_world[0], x_axis_world[1], x_axis_world[2],
              length=body_axis_length, color='red', label='Drone X-axis (Fwd)', arrow_length_ratio=0.4, linewidth=2)
    ax.quiver(drone_pos_world[0], drone_pos_world[1], drone_pos_world[2],
              y_axis_world[0], y_axis_world[1], y_axis_world[2],
              length=body_axis_length, color='green', label='Drone Y-axis (Left)', arrow_length_ratio=0.4, linewidth=2)
    ax.quiver(drone_pos_world[0], drone_pos_world[1], drone_pos_world[2],
              z_axis_world[0], z_axis_world[1], z_axis_world[2],
              length=body_axis_length, color='blue', label='Drone Z-axis (Up)', arrow_length_ratio=0.4, linewidth=2)

    # Set plot limits
    if not all_x:
        ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([0, 4])
    else:
        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        ax.set_zlim(min(all_z) - 1, max(all_z) + 1)

    ax.set_xlabel('World X-axis'); ax.set_ylabel('World Y-axis'); ax.set_zlabel('World Z-axis')
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    plt.show()


def initialize_and_visualize_reset_state(env_params, track_layout_data):
    """Initializes the environment, resets it once, and visualizes the drone state."""
    if BatchedAgiSimEnv is None:
        print("Error: BatchedAgiSimEnv is not available. Cannot initialize.")
        return

    print("Initializing BatchedAgiSimEnv for reset visualization...")
    try:
        # Using num_drones=1 for simplicity in visualizing a single reset state
        py_env = BatchedAgiSimEnv(
            sim_config_path=env_params['sim_config_path'],
            agi_param_dir=env_params['agi_param_dir'],
            sim_base_dir=env_params['sim_base_dir'],
            num_drones=1,
            track_layout=track_layout_data
        )
        # Optional: Wrap with TFPyEnvironment if your reset logic or observation parsing relies on it
        # For just getting the observation dictionary, the PyEnv might be enough if its reset() is standard.
        # eval_env = tf_py_environment.TFPyEnvironment(py_env)
        # For now, let's try with py_env directly if its reset gives the dict observation
        print("Environment initialized.")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    # Enable Roll/Yaw if your environment needs it
    # This might be specific to how your BatchedAgiSimEnv is structured.
    # If it's a direct instance, this should work. If wrapped, access the underlying env.
    if hasattr(py_env, 'enableRollYaw'):
        py_env.enableRollYaw()
        print("Roll/Yaw enabled for the environment.")

    print("Resetting the environment...")
    try:
        # The structure of time_step depends on whether you use the TFPyEnvironment wrapper
        # If using py_env directly, its reset() method should return the observation directly
        # or a structure from which we can get it.
        # Assuming reset() of BatchedAgiSimEnv (Python side) returns a list of observations
        # or if TFPyEnv wrapper is used, it returns a TimeStep object.

        # If your BatchedAgiSimEnv's reset() returns what TF-Agents expects (a TimeStep-like structure
        # or the raw observation that TFPyEnvironment would wrap), this works.
        # Let's assume py_env.reset() gives the observation directly if not TF-wrapped for simplicity.
        # For TF-Agents compatibility as in your original script, wrapping is safer.
        eval_env = tf_py_environment.TFPyEnvironment(py_env)
        time_step = eval_env.reset()

        initial_observation_dict = time_step.observation
        if not isinstance(initial_observation_dict, dict) or \
                'shared_obs' not in initial_observation_dict:
            print("Error: Initial observation is not a dictionary with 'shared_obs' key.")
            print(f"Received observation: {initial_observation_dict}")
            return

        # Assuming batch_size is 1 because num_drones=1
        shared_obs_np = initial_observation_dict['shared_obs'][0].numpy()

    except Exception as e:
        print(f"Error during environment reset or observation extraction: {e}")
        return

    # Extract and unscale position
    scaled_drone_pos = shared_obs_np[:3]
    unscaled_drone_pos = unscale_position(scaled_drone_pos, OBS_POS_MIN_NP, OBS_POS_MAX_NP)
    print(f"Drone reset to position (world): {unscaled_drone_pos}")

    # Extract body axes
    x_axis, y_axis, z_axis = get_body_axes_from_shared_obs(shared_obs_np)
    print(f"Drone X-axis (world): {x_axis}")
    print(f"Drone Y-axis (world): {y_axis}")
    print(f"Drone Z-axis (world): {z_axis}")

    # Visualize
    visualize_drone_and_track(track_layout_data, unscaled_drone_pos,
                              x_axis, y_axis, z_axis)

def main():
    env_params = {
        'sim_config_path': SIM_CONFIG_PATH,
        'agi_param_dir': AGI_PARAM_DIR,
        'sim_base_dir': SIM_BASE_DIR,
    }
    initialize_and_visualize_reset_state(env_params, TRACK_LAYOUT)

if __name__ == "__main__":
    main()