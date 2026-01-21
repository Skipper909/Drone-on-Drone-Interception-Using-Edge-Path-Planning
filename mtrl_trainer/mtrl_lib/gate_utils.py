from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import json

CURRICULUM_GATE_WIDTH = 2.0
CURRICULUM_GATE_HEIGHT = 2.0

FINAL_TRACK_LAYOUT = [
    [[7.0000, -2.0000, 2.0000], [7.0000, -2.0000, 4.0000], [7.0000, -4.0000, 4.0000], [7.0000, -4.0000, 2.0000]],
    [[9.9825, -0.0002, 2.0000], [9.9825, -0.0002, 4.0000], [12.0175, -0.0002, 4.0000], [12.0175, -0.0002, 2.0000]],
    [[7.0000, 2.0000, 3.0000], [7.0000, 2.0000, 5.0000], [7.0000, 4.0000, 5.0000], [7.0000, 4.0000, 3.0000]],
    [[-1.9825, -3.9998, 3.0000], [-1.9825, -3.9998, 5.0000], [-2.0175, -2.0002, 5.0000], [-2.0175, -2.0002, 3.0000]],
    #[[-7.0102, -0.0152, 2.0000], [-7.0102, -0.0152, 4.0000], [-5.0175, 0.0152, 4.0000], [-5.0175, 0.0152, 2.0000]],
    [[-6.9998, 0.0175, 2.0000], [-6.9998, 0.0175, 4.0000], [-5.0002, -0.0175, 4.0000], [-5.0002, -0.0175, 2.0000]],
    #[[-5.0002, 0.0175, 2.0000], [-5.0002, 0.0175, 4.0000], [-6.9998, -0.0175, 4.0000], [-6.9998, -0.0175, 2.0000]],
    [[-2.0000, 4.0000, 2.0000], [-2.0000, 4.0000, 4.0000], [-2.0000, 2.0000, 4.0000], [-2.0000, 2.0000, 2.0000]]
]

NUM_GATES = len(FINAL_TRACK_LAYOUT)

def load_track_from_file(filepath):
    """Loads a track layout from a specified JSON file."""
    with open(filepath, 'r') as f:
        track_layout = json.load(f)
    return track_layout

def get_pose_from_corners_py(corners_np):
    center = np.mean(corners_np, axis=0)
    vec_top_edge = corners_np[1] - corners_np[0]    # Towards local -X (left)
    vec_right_edge = corners_np[3] - corners_np[0]  # Towards local -Y (down)

    gate_z_axis_normal = np.cross(vec_top_edge, vec_right_edge)
    if np.linalg.norm(gate_z_axis_normal) < 1e-6:
        print("Warning: Degenerate gate from corners, using default normal [0,0,1]")
        gate_z_axis_normal = np.array([0, 0, 1]) # Fallback normal
    else:
        gate_z_axis_normal = gate_z_axis_normal / np.linalg.norm(gate_z_axis_normal)

    gate_x_axis = ((corners_np[0] - corners_np[1]) + (corners_np[3] - corners_np[2])) / 2.0
    if np.linalg.norm(gate_x_axis) < 1e-6:
        print("Warning: Could not determine gate X-axis, using fallback.")
        # Fallback: try to make it orthogonal to normal and world Z
        world_z = np.array([0,0,1])
        if np.abs(np.dot(gate_z_axis_normal, world_z)) > 0.99: # Normal is aligned with world Z
            gate_x_axis = np.array([1,0,0]) # Use world X
        else:
            gate_x_axis = np.cross(world_z, gate_z_axis_normal)
    gate_x_axis = gate_x_axis / np.linalg.norm(gate_x_axis)

    gate_y_axis = np.cross(gate_z_axis_normal, gate_x_axis)
    rotation_matrix = np.column_stack((gate_x_axis, gate_y_axis, gate_z_axis_normal))

    try:
        orientation = R.from_matrix(rotation_matrix)
    except ValueError as e:
        print(f"Error creating Rotation from matrix for gate (center: {center}): {e}")
        print(f"Matrix:\n{rotation_matrix}")
        print("Using Identity orientation as fallback.")
        orientation = R.identity()

    return {'center': center, 'orientation': orientation}

def reconstruct_corners_from_pose_py(center_np, orientation_scipy_rot, width, height):
    half_w, half_h = width / 2.0, height / 2.0
    local_corners = np.array([
        [ half_w,  half_h, 0.0],
        [-half_w,  half_h, 0.0],
        [-half_w, -half_h, 0.0],
        [ half_w, -half_h, 0.0]
    ])
    world_corners = orientation_scipy_rot.apply(local_corners) + center_np
    return world_corners.tolist() # Return as list of lists of floats

def generate_initial_easy_poses_py(num_gates, gate_width, gate_height):
    initial_poses = []
    start_x, common_y, common_z = 3.0, 0.0, 1.5
    spacing = 3.0

    gate_local_X_in_world = np.array([0., 1., 0.])  # Aligned with World Y
    gate_local_Y_in_world = np.array([0., 0., 1.])  # Aligned with World Z
    gate_local_Z_in_world = np.array([1., 0., 0.])  # Aligned with World X (This is the gate's normal)

    rotation_matrix_gate_to_world = np.column_stack((
        gate_local_X_in_world,
        gate_local_Y_in_world,
        gate_local_Z_in_world
    ))

    initial_orientation = R.from_matrix(rotation_matrix_gate_to_world)

    for i in range(num_gates):
        center = np.array([start_x + i * spacing, common_y, common_z])
        initial_poses.append({'center': center, 'orientation': initial_orientation})
    return initial_poses

def initialize_gates(track_filepath):
    final_track_layout = load_track_from_file(track_filepath)
    num_gates = len(final_track_layout)

    gate_index_to_nudge = 4
    problematic_gate_corners_original = np.array(final_track_layout[gate_index_to_nudge])

    gate_center = np.mean(problematic_gate_corners_original, axis=0)
    nudge_angle_deg = -1.0
    nudge_angle_rad = np.deg2rad(nudge_angle_deg)
    nudge_axis_world = np.array([0., 0., 1.]) # Rotate around world Z-axis

    nudge_rotation = R.from_rotvec(nudge_angle_rad * nudge_axis_world)

    nudged_gate_corners_list = []
    for corner_coords_original in problematic_gate_corners_original:
        corner_relative_to_center = corner_coords_original - gate_center
        nudged_relative_corner = nudge_rotation.apply(corner_relative_to_center)
        nudged_world_corner = nudged_relative_corner + gate_center
        nudged_gate_corners_list.append(nudged_world_corner.tolist()) # Convert back to list of lists for consistency

    final_track_layout[gate_index_to_nudge] = nudged_gate_corners_list
    final_track_poses = [get_pose_from_corners_py(np.array(gate_corners)) for gate_corners in final_track_layout]
    initial_easy_poses = generate_initial_easy_poses_py(num_gates, CURRICULUM_GATE_WIDTH, CURRICULUM_GATE_HEIGHT)

    initial_interpolated_corners = []
    for i in range(num_gates):
        corners = reconstruct_corners_from_pose_py(
            initial_easy_poses[i]['center'],
            initial_easy_poses[i]['orientation'],
            CURRICULUM_GATE_WIDTH, CURRICULUM_GATE_HEIGHT
        )
        initial_interpolated_corners.append(corners)
    return initial_easy_poses, final_track_poses, initial_interpolated_corners

