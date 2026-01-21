import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_gate_corners(center_x, center_y, center_z, side_length, roll_deg, pitch_deg, yaw_deg):
    """
    Calculates the 3D coordinates of the four corners of a square gate.

    The gate is initially defined in its local XY plane, centered at the origin,
    with its normal along the local +Z axis. It's then rotated and translated.

    Args:
        center_x (float): X-coordinate of the gate center.
        center_y (float): Y-coordinate of the gate center.
        center_z (float): Z-coordinate of the gate center.
        side_length (float): The side length of the square gate.
        roll_deg (float): Roll angle in degrees (rotation around the local X-axis after pitch and yaw).
        pitch_deg (float): Pitch angle in degrees (rotation around the local Y-axis after yaw).
        yaw_deg (float): Yaw angle in degrees (rotation around the local Z-axis first).
                         This uses an extrinsic 'ZYX' convention:
                         1. Yaw around the world Z-axis.
                         2. Pitch around the (newly rotated) world Y-axis.
                         3. Roll around the (twice rotated) world X-axis.
                         Or, equivalently, intrinsic 'xyz' (roll about local x, then pitch about new local y, then yaw about new local z).
                         For clarity, we'll use scipy's 'ZYX' which applies rotations in order Z, then Y, then X.
                         If you prefer roll, then pitch, then yaw applied to the gate's own axes (intrinsic 'xyz'):
                         Use 'xyz' for the euler_sequence and provide [roll, pitch, yaw].
                         Here, we use 'ZYX' with [yaw, pitch, roll] to match a common world-to-body or object orientation.

    Returns:
        list: A list of 4 lists, where each inner list contains the [x, y, z]
              coordinates of a gate corner. The order is:
              [top-right, top-left, bottom-left, bottom-right]
              when looking through the gate along its positive local Z-axis (normal)
              before rotation, or if the gate is aligned with world XY plane.
    """
    gate_center_world = np.array([center_x, center_y, center_z])
    half_side = side_length / 2.0

    # Define corners in the gate's local coordinate system (XY plane, normal along +Z)
    # Order: (+x,+y), (-x,+y), (-x,-y), (+x,-y)
    # This corresponds to:
    # Corner 1: Top-right
    # Corner 2: Top-left
    # Corner 3: Bottom-left
    # Corner 4: Bottom-right
    # (when looking along the gate's normal, assuming +Y is up, +X is right locally)
    local_corners = np.array([
        [half_side, half_side, 0.0],
        [-half_side, half_side, 0.0],
        [-half_side, -half_side, 0.0],
        [half_side, -half_side, 0.0]
    ])

    # Create rotation object from Euler angles.
    # Using 'ZYX' sequence (extrinsic rotations):
    # 1. Yaw rotation around Z-axis
    # 2. Pitch rotation around Y-axis (of the already Z-rotated frame)
    # 3. Roll rotation around X-axis (of the already ZY-rotated frame)
    # The angles correspond to [yaw, pitch, roll] for this sequence.
    rotation = R.from_euler('ZYX', [yaw_deg, pitch_deg, roll_deg], degrees=True)

    # Rotate the local corners
    rotated_corners = rotation.apply(local_corners)

    # Translate corners to the world position
    world_corners = rotated_corners + gate_center_world

    # Format for output
    return world_corners.tolist()


def main():
    """
    Main function to get user input and print gate corner coordinates.
    """
    print("Gate Corner Coordinate Calculator")
    print("---------------------------------")
    print("This script calculates the 4 corner coordinates of a square gate.")
    print("The gate is defined by its center, side length, and orientation (Euler angles).")
    print("Euler angle convention: ZYX extrinsic (Yaw, then Pitch, then Roll applied to world axes).")
    print("Local gate definition: Lies in XY-plane, normal along +Z, before rotation.\n")

    try:
        center_x = float(input("Enter gate center X: "))
        center_y = float(input("Enter gate center Y: "))
        center_z = float(input("Enter gate center Z: "))
        side_length = float(input("Enter gate side length (must be positive): "))
        if side_length <= 0:
            print("Error: Side length must be positive.")
            return

        yaw_deg = float(input("Enter Yaw angle in degrees (rotation around Z-axis): "))
        pitch_deg = float(input("Enter Pitch angle in degrees (rotation around Y-axis): "))
        roll_deg = float(input("Enter Roll angle in degrees (rotation around X-axis): "))

        corners = calculate_gate_corners(center_x, center_y, center_z, side_length, roll_deg, pitch_deg, yaw_deg)

        print("\nCalculated Gate Corners (format: [[x1,y1,z1], [x2,y2,z2], ...]):")
        # For direct use in your track_layout format:
        print("[")
        for i, corner in enumerate(corners):
            print(f"    [{corner[0]:.4f}, {corner[1]:.4f}, {corner[2]:.4f}]", end="")
            if i < len(corners) - 1:
                print(",")
            else:
                print()
        print("]")

        # Example of how to make it a single line for easy copy-paste
        corners_str = ", ".join([f"[{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]" for c in corners])
        print(f"\nAs a single line for track_layout: \n[[{corners_str}]]")


    except ValueError:
        print("Error: Invalid input. Please enter numerical values.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
