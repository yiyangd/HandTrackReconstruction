import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pdb
import open3d as o3d
from hand_landmark_estimation import estimate_hand_orientation
from scipy.spatial.transform import Rotation as sR

def Randt2T(R, t=None):
    """
    Converts a rotation matrix and translation vector into a 4x4 transformation matrix.

    Parameters:
    - R: 3x3 rotation matrix
    - t: 3x1 translation vector (optional)

    Returns:
    - T: 4x4 transformation matrix combining R and t
    """
    T = np.eye(4)
    T[:3, :3] = R  # Set rotation part
    if t is not None:
        t = t.reshape(-1)  # Ensure translation is 1D
        T[:3, 3] = t[:3]  # Set translation part
    return T


def Angle2RotationMatrix(angles, l):
    """
    Converts Euler angles to a rotation matrix, optionally returning a 4x4 transformation matrix.

    Parameters:
    - angles: tuple or list of 3 Euler angles (roll, pitch, yaw)
    - l: integer indicating the output format (3 for 3x3 rotation matrix, 4 for 4x4 transformation matrix)

    Returns:
    - R: 3x3 rotation matrix or 4x4 transformation matrix if l == 4
    """
    r, p, y = angles
    # Rotation matrix around x-axis
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    # Rotation matrix around y-axis
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    # Rotation matrix around z-axis
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    if l == 4:
        R = Randt2T(R)  # Return 4x4 transformation matrix if l == 4
    return R


def point2pixel(point):
    """
    Projects a 3D point to a 2D pixel using predefined camera intrinsics.

    Parameters:
    - point: 3x1 3D point in camera coordinates
    p_c_0 = [-0.20990816, -0.11388494,  0.54248083,  1.        ]



    Returns:
    - pixel: 2x1 pixel coordinates (u, v)
    uv_0 = [88, 120]
    """
    point = point.reshape(-1)
    # Camera intrinsic parameters (focal lengths and principal point)
    intrinsics = np.array([[609.62, 0, 323.44],  # fx cx
                           [0, 609.67, 247.87],  # fy cy
                           [0, 0, 1]])
    x = point[0] / point[2]  # Normalized x-coordinate
    y = point[1] / point[2]  # Normalized y-coordinate
    # Compute pixel u-coordinate
    u = np.round(x * intrinsics[0, 0] + intrinsics[0, 2])
    # Compute pixel v-coordinate
    v = np.round(y * intrinsics[1, 1] + intrinsics[1, 2])
    pixel = np.array([u, v])
    return pixel


def print_pred_results(x):
    """
    Prints the predicted results, such as joint angles and bending angles for hand joints.

    Parameters:
    - x: 1D array containing the model parameters, including joint angles and bending angles
    """
    angles0 = x[:3]  # Landmark 0 angles (roll, pitch, yaw)
    print(f'landmark 0, roll: {angles0[0]:.2f}, pitch: {angles0[1]:.2f}  yow: {angles0[2]:.2f}')
    # Print angles for other landmarks (1, 5, 9, 13, 17)
    angles1 = x[3:6]
    angles5 = x[6:9]
    angles9 = x[9:12]
    angles13 = x[12:15]
    angles17 = x[15:18]
    print(f'landmark 1, roll: {angles1[0]:.2f}, pitch: {angles1[1]:.2f}  yow: {angles1[2]:.2f}')
    print(f'landmark 5, roll: {angles5[0]:.2f}, pitch: {angles5[1]:.2f}  yow: {angles5[2]:.2f}')
    print(f'landmark 9, roll: {angles9[0]:.2f}, pitch: {angles9[1]:.2f}  yow: {angles9[2]:.2f}')
    print(f'landmark 13, roll: {angles13[0]:.2f}, pitch: {angles13[1]:.2f}  yow: {angles13[2]:.2f}')
    print(f'landmark 17, roll: {angles17[0]:.2f}, pitch: {angles17[1]:.2f}  yow: {angles17[2]:.2f}')
    # Bending angles for fingers (beta values)
    beta2, beta3 = x[18:20]
    beta6, beta7 = x[20:22]
    beta10, beta11 = x[22:24]
    beta14, beta15 = x[24:26]
    beta18, beta19 = x[26:28]

    # Flexion angles (theta values) for fingers
    theta1, theta5, theta9, theta13, theta17 = x[28: 33]
    print(f"Thumb bending angle: {theta1 * 180 / np.pi:.2f},{beta2 * 180 / np.pi:.2f} 和 {beta3 * 180 / np.pi:.2f}")
    print(
        f"Index finger bending angle: {theta5 * 180 / np.pi:.2f},{beta6 * 180 / np.pi:.2f} 和 {beta7 * 180 / np.pi:.2f}")
    print(
        f"Middle finger bending angle: {theta9 * 180 / np.pi:.2f},{beta10 * 180 / np.pi:.2f} 和 {beta11 * 180 / np.pi:.2f}")
    print(
        f"Ring finger bending angle: {theta13 * 180 / np.pi:.2f},{beta14 * 180 / np.pi:.2f} 和 {beta15 * 180 / np.pi:.2f}")
    print(
        f"Pinky finger bending angle: {theta17 * 180 / np.pi:.2f},{beta18 * 180 / np.pi:.2f} 和 {beta19 * 180 / np.pi:.2f}")

    q0 = x[33:36]  # Base position of the hand (x, y, z)


def parse_parameters(x):
    """
    Parses the parameter array into individual components representing joint angles, bending angles, and hand position.

    Parameters:
    - x: 1D array containing model parameters

    Returns:
    - Tuple containing the parsed parameters:
        - angles for joints
        - bending angles (beta values)
        - flexion angles (theta values)
        - hand position (q0)
    """
    angles0 = x[:3]

    angles1 = x[3:5]
    angles5 = x[5:7]
    angles9 = x[7:9]
    angles13 = x[9:11]
    angles17 = x[11:13]

    beta2, beta3 = x[13:15]
    beta6, beta7 = x[15:17]
    beta10, beta11 = x[17:19]
    beta14, beta15 = x[19:21]
    beta18, beta19 = x[21:23]

    theta1, theta5, theta9, theta13, theta17 = x[23: 28]

    q0 = x[28:31]
    return (angles0, angles1, angles5, angles9, angles13, angles17,
            beta2, beta3, beta6, beta7, beta10, beta11, beta14, beta15, beta18, beta19,
            theta1, theta5, theta9, theta13, theta17,
            q0)


def fit_last_three_landmarks(beta, t, length, T_pre):
    """
    Computes the transformation and pixel coordinates for the last three hand landmarks.

    Parameters:
    - beta: Joint bending angle between the current and previous landmarks.
    - t: 3D translation vector from the previous landmark.
    - length: Length of the bone segment connecting the landmarks.
    - T_pre: Transformation matrix of the previous landmark.

    Returns:
    - T_2_3: Transformation matrix from the current landmark to the next.
    - p_c_4: Transformed coordinates of the next landmark in world coordinates.
    - p_3_4: Local coordinates of the next landmark in the local frame.
    - uv_4: Pixel coordinates of the next landmark in the image plane.
    """
    R_2_3 = Angle2RotationMatrix(np.array([0, beta, 0]), 3)  # Rotation matrix based on joint bending angle.
    t_2_3 = t
    T_2_3 = Randt2T(R_2_3, t_2_3)  # Transformation matrix combining rotation and translation.
    p_3_4 = np.array([[length, 0, 0, 1]])  # Point of the next landmark in local coordinates.
    p_c_4 = (T_pre @ T_2_3 @ p_3_4.T).T  # Transform the point to world coordinates.
    uv_4 = point2pixel(p_c_4)  # Project the world coordinates to pixel coordinates.
    return T_2_3, p_c_4, p_3_4, uv_4


def get_global_rotation_matrices(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
                                 T_10_11, T_13_14, T_14_15, T_17_18, T_18_19):
    """
    Computes global transformation matrices for all hand landmarks, starting from the palm base.

    Parameters:
    - T_c_0: Base transformation matrix of the hand's center.
    - T_0_1, T_0_5, T_0_9, T_0_13, T_0_17: Transformation matrices for the joints.
    - T_1_2, T_2_3, T_5_6, T_6_7, T_9_10, T_10_11, T_13_14, T_14_15, T_17_18, T_18_19: Additional joint transformation matrices.

    Returns:
    - global_frames: Dictionary of transformation matrices for all landmarks.
    """
    T_c_1 = T_c_0 @ T_0_1  # Transform from center to landmark 1.
    T_c_2 = T_c_1 @ T_1_2  # Transform from landmark 1 to landmark 2.
    T_c_3 = T_c_2 @ T_2_3  # Continue transformation to landmark 3.
    # Similar transformations for the other fingers (5, 9, 13, 17).
    T_c_5 = T_c_0 @ T_0_5
    T_c_6 = T_c_5 @ T_5_6
    T_c_7 = T_c_6 @ T_6_7

    T_c_9 = T_c_0 @ T_0_9
    T_c_10 = T_c_9 @ T_9_10
    T_c_11 = T_c_10 @ T_10_11

    T_c_13 = T_c_0 @ T_0_13
    T_c_14 = T_c_13 @ T_13_14
    T_c_15 = T_c_14 @ T_14_15

    T_c_17 = T_c_0 @ T_0_17
    T_c_18 = T_c_17 @ T_17_18
    T_c_19 = T_c_18 @ T_18_19

    global_frames = {'T_c_0': T_c_0, 'T_c_1': T_c_1, 'T_c_2': T_c_2, 'T_c_3': T_c_3,
                     'T_c_5': T_c_5, 'T_c_6': T_c_6, 'T_c_7': T_c_7, 'T_c_9': T_c_9,
                     'T_c_10': T_c_10, 'T_c_11': T_c_11, 'T_c_13': T_c_13, 'T_c_14': T_c_14,
                     'T_c_15': T_c_15, 'T_c_17': T_c_17, 'T_c_18': T_c_18, 'T_c_19': T_c_19}
    return global_frames


def foward_hand_model(x, lengths):  # x is the parameter to be estimated, p is the observed values
    """
    Simulates forward kinematics for a hand model, estimating landmark positions and pixel coordinates.

    Parameters:
    - x: Model parameters to be estimated, including joint angles and bending angles.
    - lengths: Bone segment lengths for each finger.

    Returns:
    - pred_uv: Predicted pixel coordinates for each hand landmark.
    - pred_p: Predicted 3D positions for each hand landmark.
    - global_frames: Transformation matrices for all landmarks.
    - data_list: List of rotation matrices for all landmarks.
    """
    (angles0, angles1, angles5, angles9, angles13, angles17,
     beta2, beta3, beta6, beta7, beta10, beta11, beta14, beta15, beta18, beta19,
     theta1, theta5, theta9, theta13, theta17, q0) = parse_parameters(x)

    p_c_0 = np.array([q0[0], q0[1], q0[2], 1])  # Base landmark 0 in 3D.
    uv_0 = point2pixel(p_c_0)  # Project to pixel coordinates.

    R_c_0 = Angle2RotationMatrix(angles0, 3)
    t_c_0 = p_c_0
    T_c_0 = Randt2T(R_c_0, t_c_0)

    tempR1 = Angle2RotationMatrix(np.array([0, 0, theta1]), 4)
    p_0_1 = (tempR1 @ np.array([[lengths[0, 0], 0, 0, 1]]).T).T  # l01
    p_c_1 = (T_c_0 @ p_0_1.T).T
    uv_1 = point2pixel(p_c_1)

    R_0_1 = Angle2RotationMatrix(np.array([0, angles1[0], angles1[1]]), 3)
    t_0_1 = p_0_1
    T_0_1 = Randt2T(R_0_1, t_0_1)
    p_1_2 = np.array([[lengths[0, 1], 0, 0, 1]])  # l12
    p_c_2 = (T_c_0 @ T_0_1 @ p_1_2.T).T
    uv_2 = point2pixel(p_c_2)

    T_1_2, p_c_3, p_2_3, uv_3 = fit_last_three_landmarks(beta2, p_1_2, lengths[0, 2], T_c_0 @ T_0_1)
    T_2_3, p_c_4, p_3_4, uv_4 = fit_last_three_landmarks(beta3, p_2_3, lengths[0, 3], T_c_0 @ T_0_1 @ T_1_2)

    tempR5 = Angle2RotationMatrix(np.array([0, 0, theta5]), 4)
    p_0_5 = (tempR5 @ np.array([[lengths[1, 0], 0, 0, 1]]).T).T  # l01
    p_c_5 = (T_c_0 @ p_0_5.T).T
    uv_5 = point2pixel(p_c_5)

    R_0_5 = Angle2RotationMatrix(np.array([0, angles5[0], angles5[1]]), 3)
    t_0_5 = p_0_5
    T_0_5 = Randt2T(R_0_5, t_0_5)
    p_5_6 = np.array([[lengths[1, 1], 0, 0, 1]])  # l12
    p_c_6 = (T_c_0 @ T_0_5 @ p_5_6.T).T
    uv_6 = point2pixel(p_c_6)

    T_5_6, p_c_7, p_6_7, uv_7 = fit_last_three_landmarks(beta6, p_5_6, lengths[1, 2], T_c_0 @ T_0_5)
    T_6_7, p_c_8, p_7_8, uv_8 = fit_last_three_landmarks(beta7, p_6_7, lengths[1, 3], T_c_0 @ T_0_5 @ T_5_6)

    tempR9 = Angle2RotationMatrix(np.array([0, 0, theta9]), 4)
    p_0_9 = (tempR9 @ np.array([[lengths[2, 0], 0, 0, 1]]).T).T  # l01
    p_c_9 = (T_c_0 @ p_0_9.T).T
    uv_9 = point2pixel(p_c_9)

    R_0_9 = Angle2RotationMatrix(np.array([0, angles9[0], angles9[1]]), 3)
    t_0_9 = p_0_9
    T_0_9 = Randt2T(R_0_9, t_0_9)
    p_9_10 = np.array([[lengths[2, 1], 0, 0, 1]])  # l12
    p_c_10 = (T_c_0 @ T_0_9 @ p_9_10.T).T
    uv_10 = point2pixel(p_c_10)

    T_9_10, p_c_11, p_10_11, uv_11 = fit_last_three_landmarks(beta10, p_9_10, lengths[2, 2], T_c_0 @ T_0_9)
    T_10_11, p_c_12, p_11_12, uv_12 = fit_last_three_landmarks(beta11, p_10_11, lengths[2, 3], T_c_0 @ T_0_9 @ T_9_10)

    tempR13 = Angle2RotationMatrix(np.array([0, 0, theta13]), 4)
    p_0_13 = (tempR13 @ np.array([[lengths[3, 0], 0, 0, 1]]).T).T  # l01
    p_c_13 = (T_c_0 @ p_0_13.T).T
    uv_13 = point2pixel(p_c_13)

    R_0_13 = Angle2RotationMatrix(np.array([0, angles13[0], angles13[1]]), 3)
    t_0_13 = p_0_13
    T_0_13 = Randt2T(R_0_13, t_0_13)
    p_13_14 = np.array([[lengths[3, 1], 0, 0, 1]])  # l12
    p_c_14 = (T_c_0 @ T_0_13 @ p_13_14.T).T
    uv_14 = point2pixel(p_c_14)

    T_13_14, p_c_15, p_14_15, uv_15 = fit_last_three_landmarks(beta14, p_13_14, lengths[3, 2], T_c_0 @ T_0_13)
    T_14_15, p_c_16, p_15_16, uv_16 = fit_last_three_landmarks(beta15, p_14_15, lengths[3, 3], T_c_0 @ T_0_13 @ T_13_14)

    tempR17 = Angle2RotationMatrix(np.array([0, 0, theta17]), 4)
    p_0_17 = (tempR17 @ np.array([[lengths[4, 0], 0, 0, 1]]).T).T  # l01
    p_c_17 = (T_c_0 @ p_0_17.T).T
    uv_17 = point2pixel(p_c_17)

    R_0_17 = Angle2RotationMatrix(np.array([0, angles17[0], angles17[1]]), 3)
    t_0_17 = p_0_17
    T_0_17 = Randt2T(R_0_17, t_0_17)
    p_17_18 = np.array([[lengths[4, 1], 0, 0, 1]])  # l12
    p_c_18 = (T_c_0 @ T_0_17 @ p_17_18.T).T
    uv_18 = point2pixel(p_c_18)

    T_17_18, p_c_19, p_18_19, uv_19 = fit_last_three_landmarks(beta18, p_17_18, lengths[4, 2], T_c_0 @ T_0_17)
    T_18_19, p_c_20, p_19_20, uv_20 = fit_last_three_landmarks(beta19, p_18_19, lengths[4, 3], T_c_0 @ T_0_17 @ T_17_18)

    pred_uv = np.stack([uv_0, uv_1, uv_2, uv_3, uv_4, uv_5, uv_6, uv_7, uv_8, uv_9, uv_10,
                        uv_11, uv_12, uv_13, uv_14, uv_15, uv_16, uv_17, uv_18, uv_19, uv_20], 0)

    pred_p = np.concatenate([p_c_0[None], p_c_1, p_c_2, p_c_3, p_c_4, p_c_5, p_c_6, p_c_7, p_c_8, p_c_9, p_c_10,
                             p_c_11, p_c_12, p_c_13, p_c_14, p_c_15, p_c_16, p_c_17, p_c_18, p_c_19, p_c_20], 0)

    global_frames = get_global_rotation_matrices(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7,
                                                 T_9_10,
                                                 T_10_11, T_13_14, T_14_15, T_17_18, T_18_19)
    # print(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
    #                                  T_10_11, T_13_14, T_14_15, T_17_18, T_18_19)

    # Get all rotation matrices.
    data_list = local_frame_to_list(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
                                    T_10_11, T_13_14, T_14_15, T_17_18, T_18_19)
    # print(global_frames)
    # import pdb; pdb.set_trace()
    return pred_uv, pred_p, global_frames, data_list


def local_frame_to_list(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
                        T_10_11, T_13_14, T_14_15, T_17_18, T_18_19):
    """
    Converts transformation matrices into a flat list of rotation matrices for all hand landmarks.

    Parameters:
    - Transformation matrices for all hand landmarks.

    Returns:
    - data_list: Flattened list of 3x3 rotation matrices for all landmarks.
    """
    data_list = []
    # Rotation matrix for 180-degree rotation around the z-axis
    #
    rotation_180_z = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    rotation_180_y = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    # # Apply rotations to each transformation matrix and append the resulting rotation matrices to the list.
    # R_c_0 = rotation_180_y @ T_c_0[:3, :3] @ rotation_180_y

    R_c_0 = rotation_180_z @ T_c_0[:3, :3] @ rotation_180_z
    R_c_0 = Angle2RotationMatrix([np.pi, 0, 0], 3) @ R_c_0
    T_0_1[:3, :3] = rotation_180_z @ T_0_1[:3, :3] @ rotation_180_z
    T_1_2[:3, :3] = rotation_180_z @ T_1_2[:3, :3] @ rotation_180_z
    T_2_3[:3, :3] = rotation_180_z @ T_2_3[:3, :3] @ rotation_180_z
    T_0_5[:3, :3] = rotation_180_z @ T_0_5[:3, :3] @ rotation_180_z
    T_5_6[:3, :3] = rotation_180_z @ T_5_6[:3, :3] @ rotation_180_z
    T_6_7[:3, :3] = rotation_180_z @ T_6_7[:3, :3] @ rotation_180_z
    T_0_9[:3, :3] = rotation_180_z @ T_0_9[:3, :3] @ rotation_180_z
    T_9_10[:3, :3] = rotation_180_z @ T_9_10[:3, :3] @ rotation_180_z
    T_10_11[:3, :3] = rotation_180_z @ T_10_11[:3, :3] @ rotation_180_z
    T_0_13[:3, :3] = rotation_180_z @ T_0_13[:3, :3] @ rotation_180_z
    T_13_14[:3, :3] = rotation_180_z @ T_13_14[:3, :3] @ rotation_180_z
    T_14_15[:3, :3] = rotation_180_z @ T_14_15[:3, :3] @ rotation_180_z
    T_0_17[:3, :3] = rotation_180_z @ T_0_17[:3, :3] @ rotation_180_z
    T_17_18[:3, :3] = rotation_180_z @ T_17_18[:3, :3] @ rotation_180_z
    T_18_19[:3, :3] = rotation_180_z @ T_18_19[:3, :3] @ rotation_180_z

    data_list.extend(R_c_0.reshape(-1).tolist())
    data_list.extend(T_0_1[:3, :3].reshape(-1).tolist())
    data_list.extend(T_0_5[:3, :3].reshape(-1).tolist())
    data_list.extend(T_0_9[:3, :3].reshape(-1).tolist())
    data_list.extend(T_0_13[:3, :3].reshape(-1).tolist())
    data_list.extend(T_0_17[:3, :3].reshape(-1).tolist())
    data_list.extend(T_1_2[:3, :3].reshape(-1).tolist())
    data_list.extend(T_2_3[:3, :3].reshape(-1).tolist())
    data_list.extend(T_5_6[:3, :3].reshape(-1).tolist())
    data_list.extend(T_6_7[:3, :3].reshape(-1).tolist())
    data_list.extend(T_9_10[:3, :3].reshape(-1).tolist())
    data_list.extend(T_10_11[:3, :3].reshape(-1).tolist())
    data_list.extend(T_13_14[:3, :3].reshape(-1).tolist())
    data_list.extend(T_14_15[:3, :3].reshape(-1).tolist())
    data_list.extend(T_17_18[:3, :3].reshape(-1).tolist())
    data_list.extend(T_18_19[:3, :3].reshape(-1).tolist())
    return data_list


# def local_frame_to_list(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
#                                      T_10_11, T_13_14, T_14_15, T_17_18, T_18_19):
#     data_list = []
#     # Rotation matrix for 180-degree rotation around the z-axis
#     rotation_180_z = np.array([
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, 0, 1]
#     ])
#     rotation_180_y = np.array([
#         [-1, 0, 0],
#         [0, 1, 0],
#         [0, 0, -1]
#     ])
#     # T_c_0[:, 2] = - T_c_0[:, 2]
#     R_c_0 = np.dot(rotation_180_z, T_c_0[:3, :3])
#     print(R_c_0.reshape(-1))
#     # T_0_1[:, 1] = - T_0_1[:, 1]
#     R_0_5 = np.dot(rotation_180_z, T_0_5[:3, :3])
#     # T_0_9[:, 1] = - T_0_9[:, 1]
#     # T_0_13[:, 1] = - T_0_13[:, 1]
#     # T_0_17[:, 1] = - T_0_17[:, 1]

#     # T_1_2[:, 1] = - T_1_2[:, 1]
#     # T_2_3[:, 1] = - T_2_3[:, 1]
#     R_5_6 = np.dot(rotation_180_z, T_5_6[:3, :3])
#     R_6_7 = np.dot(rotation_180_z, T_6_7[:3, :3])
#     # T_9_10[:, 1] = - T_9_10[:, 1]
#     # T_10_11[:, 1] = - T_10_11[:, 1]
#     # T_13_14[:, 1] = - T_13_14[:, 1]
#     # T_14_15[:, 1] = - T_14_15[:, 1]
#     # T_17_18[:, 1] = - T_17_18[:, 1]
#     # T_18_19[:, 1] = - T_18_19[:, 1]

#     data_list.extend(R_c_0.reshape(-1).tolist())
#     data_list.extend(T_0_1[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_0_5[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_0_9[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_0_13[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_0_17[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_1_2[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_2_3[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_5_6[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_6_7[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_9_10[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_10_11[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_13_14[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_14_15[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_17_18[:3, :3].reshape(-1).tolist())
#     data_list.extend(T_18_19[:3, :3].reshape(-1).tolist())
#     return data_list

# def compute_para_error(R_c_0, p, global_frames):
# y_pos = (p[17] - p[5]) / np.linalg.norm(p[17] - p[5], 2, -1)
# y_pred = R_c_0[:, 1]
# similarity_y = y_pos @ y_pred.T
# x_pos = (p[13] - p[0]) / np.linalg.norm(p[13] - p[0], 2, -1)
# x_pred = R_c_0[:, 0]
# similarity_x = x_pos @ x_pred.T
# return 2 - similarity_x - similarity_y
def compute_para_error(global_frames):
    """
    Computes the parameter error by comparing the alignment
    of key axis directions between hand landmarks.

    Parameters:
    - global_frames: A dictionary containing the transformation matrices for various hand landmarks.

    Returns:
    - Total error that penalizes misalignment between key axis directions of different landmarks.
    - The error includes the following components:
        - dist_5_9: Alignment error between the y-axes of landmarks 5 and 9 (middle and index fingers).
        - dist_13_17: Alignment error between the y-axes of landmarks 13 and 17 (ring and pinky fingers).
        - dist_9_13: Alignment error between the y-axes of landmarks 9 and 13 (index and ring fingers).
        - y_1_d, y_5_d, y_9_d, y_13_d, y_17_d: Alignment error between the y-axis of the palm
                                                (landmark 0) and the respective fingers.

    The error is calculated based on the dot product between the y-axes (second column of the transformation matrix) of the respective landmarks.
    A small threshold is used to penalize slight misalignments, and the final error is scaled by a small factor.
    """
    T_c_0 = global_frames['T_c_0']  # Transformation of the palm base (landmark 0).
    T_c_1 = global_frames['T_c_1']  # Transformation of landmark 1 (thumb base).
    T_c_5 = global_frames['T_c_5']  # Transformation of landmark 5 (index finger).
    T_c_9 = global_frames['T_c_9']  # Transformation of landmark 9 (middle finger).
    T_c_13 = global_frames['T_c_13']  # Transformation of landmark 13 (ring finger).
    T_c_17 = global_frames['T_c_17']  # Transformation of landmark 17 (pinky finger).

    y_0 = T_c_0[:, 1]  # y-axis direction of the palm base.
    y_1 = T_c_1[:, 1]  # y-axis direction of the thumb.
    y_5 = T_c_5[:, 1]  # y-axis direction of the index finger
    y_9 = T_c_9[:, 1]  # y-axis direction of the middle finger.
    y_13 = T_c_13[:, 1]  # y-axis direction of the ring finger.
    y_17 = T_c_17[:, 1]  # y-axis direction of the pinky finger.

    # Compute the alignment errors between key fingers (dot product with penalization).
    dist_5_9 = np.clip(1 - np.sum(y_5 * y_9) - 0.05, a_min=0, a_max=2)
    dist_13_17 = np.clip(1 - np.sum(y_13 * y_17) - 0.05, a_min=0, a_max=2)
    dist_9_13 = np.clip(1 - np.sum(y_13 * y_9) - 0.05, a_min=0, a_max=2)

    # Compute misalignment penalties between the palm base (landmark 0) and each finger.
    y_1_d = 1 - (np.sum(y_0 * y_1) > 0).astype(np.float32)
    y_5_d = 1 - (np.sum(y_0 * y_5) > 0).astype(np.float32)
    y_9_d = 1 - (np.sum(y_0 * y_9) > 0).astype(np.float32)
    y_17_d = 1 - (np.sum(y_0 * y_17) > 0).astype(np.float32)
    y_13_d = 1 - (np.sum(y_0 * y_13) > 0).astype(np.float32)
    # Return the total error including the alignment errors and penalties for misalignment.
    return dist_5_9 + dist_9_13 + dist_13_17 + (y_13_d + y_17_d + y_9_d + y_5_d) * 1e-3


def fit_hand_model(x, p, uv, lengths, outliers):  # x is the parameter to be estimated, p is the observed values
    """
    Estimates the fit of a kinematic hand model to the observed hand landmarks by minimizing the error between
    predicted and observed landmark positions and pixel coordinates.

    Parameters:
    - x: 1D numpy array, the parameters to be estimated.
        These include:
        - Joint angles for the palm and fingers.
        - Flexion angles (beta values) for the intermediate joints of the fingers.
        - Theta values for finger bending.
        - 3D position of the hand in the camera coordinate system (q0).

    - p: 2D numpy array, observed 3D positions of the hand landmarks (e.g., joints) in world space.

    - uv: 2D numpy array, observed 2D pixel coordinates of the hand landmarks in the image space.

    - lengths: 2D numpy array, length values for each bone segment of the hand. Each row corresponds to a finger,
      and columns represent the lengths of the finger segments (phalanges).

    - outliers: 1D numpy array, binary values indicating whether a landmark is considered an outlier (1) or not (0).
      These outliers may be excluded from certain computations (such as error minimization).

    Returns:
    - err: Scalar value representing the total error between the predicted and observed landmarks. This error
      is a combination of:
        - `p_err`: The difference between predicted and observed 3D landmark positions.
        - `uv_err`: The difference between predicted and observed 2D pixel positions.
        - `para_err`: The alignment error between key axes (computed via `compute_para_error`).
    """
    # Parse the input parameters (joint angles, flexion angles, bending angles, and 3D hand position)
    (angles0, angles1, angles5, angles9, angles13, angles17,
     beta2, beta3, beta6, beta7, beta10, beta11, beta14, beta15, beta18, beta19,
     theta1, theta5, theta9, theta13, theta17, q0) = parse_parameters(x)
    # Compute the 3D position of the root of the hand (palm center) and its projection to pixel coordinates
    p_c_0 = np.array([q0[0], q0[1], q0[2], 1])
    uv_0 = point2pixel(p_c_0)
    # Calculate the transformation matrix for the palm (root) using the angles and the 3D position
    R_c_0 = Angle2RotationMatrix(angles0, 3)
    t_c_0 = p_c_0
    T_c_0 = Randt2T(R_c_0, t_c_0)

    # Calculate the transformation and projected position for the thumb (first finger)
    tempR1 = Angle2RotationMatrix(np.array([0, 0, theta1]), 4)
    p_0_1 = (tempR1 @ np.array([[lengths[0, 0], 0, 0, 1]]).T).T  # Thumb's first segment (l01)
    p_c_1 = (T_c_0 @ p_0_1.T).T  # Apply palm's transformation to get thumb's 3D position
    uv_1 = point2pixel(p_c_1)  # Project thumb's 3D position into 2D pixel coordinates

    # Compute the transformation for the rest of the thumb joints
    R_0_1 = Angle2RotationMatrix(np.array([0, angles1[0], angles1[1]]), 3)
    t_0_1 = p_0_1
    T_0_1 = Randt2T(R_0_1, t_0_1)

    # Calculate and transform thumb's second joint
    p_1_2 = np.array([[lengths[0, 1], 0, 0, 1]])  # Second segment of the thumb (l12)
    p_c_2 = (T_c_0 @ T_0_1 @ p_1_2.T).T  # Apply transformations to get thumb's second joint position
    uv_2 = point2pixel(p_c_2)  # Project into 2D pixel coordinates

    # Compute the last two joints of the thumb using flexion angles (beta2, beta3)
    T_1_2, p_c_3, p_2_3, uv_3 = fit_last_three_landmarks(beta2, p_1_2, lengths[0, 2], T_c_0 @ T_0_1)
    T_2_3, p_c_4, p_3_4, uv_4 = fit_last_three_landmarks(beta3, p_2_3, lengths[0, 3], T_c_0 @ T_0_1 @ T_1_2)

    # Repeat similar computations for the other fingers (index, middle, ring, and pinky fingers)
    # For the index finger
    tempR5 = Angle2RotationMatrix(np.array([0, 0, theta5]), 4)
    p_0_5 = (tempR5 @ np.array([[lengths[1, 0], 0, 0, 1]]).T).T  # l01
    p_c_5 = (T_c_0 @ p_0_5.T).T
    uv_5 = point2pixel(p_c_5)

    R_0_5 = Angle2RotationMatrix(np.array([0, angles5[0], angles5[1]]), 3)
    t_0_5 = p_0_5
    T_0_5 = Randt2T(R_0_5, t_0_5)
    p_5_6 = np.array([[lengths[1, 1], 0, 0, 1]])  # l12
    p_c_6 = (T_c_0 @ T_0_5 @ p_5_6.T).T
    uv_6 = point2pixel(p_c_6)

    T_5_6, p_c_7, p_6_7, uv_7 = fit_last_three_landmarks(beta6, p_5_6, lengths[1, 2], T_c_0 @ T_0_5)
    T_6_7, p_c_8, p_7_8, uv_8 = fit_last_three_landmarks(beta7, p_6_7, lengths[1, 3], T_c_0 @ T_0_5 @ T_5_6)

    # For the middle finger
    tempR9 = Angle2RotationMatrix(np.array([0, 0, theta9]), 4)
    p_0_9 = (tempR9 @ np.array([[lengths[2, 0], 0, 0, 1]]).T).T  # l01
    p_c_9 = (T_c_0 @ p_0_9.T).T
    uv_9 = point2pixel(p_c_9)

    R_0_9 = Angle2RotationMatrix(np.array([0, angles9[0], angles9[1]]), 3)
    t_0_9 = p_0_9
    T_0_9 = Randt2T(R_0_9, t_0_9)
    p_9_10 = np.array([[lengths[2, 1], 0, 0, 1]])  # l12
    p_c_10 = (T_c_0 @ T_0_9 @ p_9_10.T).T
    uv_10 = point2pixel(p_c_10)

    T_9_10, p_c_11, p_10_11, uv_11 = fit_last_three_landmarks(beta10, p_9_10, lengths[2, 2], T_c_0 @ T_0_9)
    T_10_11, p_c_12, p_11_12, uv_12 = fit_last_three_landmarks(beta11, p_10_11, lengths[2, 3], T_c_0 @ T_0_9 @ T_9_10)

    # For the ring finger
    tempR13 = Angle2RotationMatrix(np.array([0, 0, theta13]), 4)
    p_0_13 = (tempR13 @ np.array([[lengths[3, 0], 0, 0, 1]]).T).T  # l01
    p_c_13 = (T_c_0 @ p_0_13.T).T
    uv_13 = point2pixel(p_c_13)

    R_0_13 = Angle2RotationMatrix(np.array([0, angles13[0], angles13[1]]), 3)
    t_0_13 = p_0_13
    T_0_13 = Randt2T(R_0_13, t_0_13)
    p_13_14 = np.array([[lengths[3, 1], 0, 0, 1]])  # l12
    p_c_14 = (T_c_0 @ T_0_13 @ p_13_14.T).T
    uv_14 = point2pixel(p_c_14)

    T_13_14, p_c_15, p_14_15, uv_15 = fit_last_three_landmarks(beta14, p_13_14, lengths[3, 2], T_c_0 @ T_0_13)
    T_14_15, p_c_16, p_15_16, uv_16 = fit_last_three_landmarks(beta15, p_14_15, lengths[3, 3], T_c_0 @ T_0_13 @ T_13_14)

    # For the pinky finger
    tempR17 = Angle2RotationMatrix(np.array([0, 0, theta17]), 4)
    p_0_17 = (tempR17 @ np.array([[lengths[4, 0], 0, 0, 1]]).T).T  # l01
    p_c_17 = (T_c_0 @ p_0_17.T).T
    uv_17 = point2pixel(p_c_17)

    R_0_17 = Angle2RotationMatrix(np.array([0, angles17[0], angles17[1]]), 3)
    t_0_17 = p_0_17
    T_0_17 = Randt2T(R_0_17, t_0_17)
    p_17_18 = np.array([[lengths[4, 1], 0, 0, 1]])  # l12
    p_c_18 = (T_c_0 @ T_0_17 @ p_17_18.T).T
    uv_18 = point2pixel(p_c_18)

    T_17_18, p_c_19, p_18_19, uv_19 = fit_last_three_landmarks(beta18, p_17_18, lengths[4, 2], T_c_0 @ T_0_17)
    T_18_19, p_c_20, p_19_20, uv_20 = fit_last_three_landmarks(beta19, p_18_19, lengths[4, 3], T_c_0 @ T_0_17 @ T_17_18)

    # Compute the global rotation matrices for each joint (for error computation)
    global_frames = get_global_rotation_matrices(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7,
                                                 T_9_10,
                                                 T_10_11, T_13_14, T_14_15, T_17_18, T_18_19)

    pred_uv = np.stack([uv_0, uv_1, uv_2, uv_3, uv_4, uv_5, uv_6, uv_7, uv_8, uv_9, uv_10,
                        uv_11, uv_12, uv_13, uv_14, uv_15, uv_16, uv_17, uv_18, uv_19, uv_20], 0)

    pred_p = np.concatenate([p_c_0[None], p_c_1, p_c_2, p_c_3, p_c_4, p_c_5, p_c_6, p_c_7, p_c_8, p_c_9, p_c_10,
                             p_c_11, p_c_12, p_c_13, p_c_14, p_c_15, p_c_16, p_c_17, p_c_18, p_c_19, p_c_20], 0)

    uv_err = np.sqrt(np.sum((pred_uv - uv) ** 2, -1))
    p_err = np.sqrt(np.sum((pred_p[:, :3] - p) ** 2, -1))
    para_err = compute_para_error(global_frames)

    # err = np.sum(p_err * (1 - outliers) + 1e-5 * uv_err * outliers) + para_err * 1e-3
    err = np.sum(p_err * (1 - outliers) + 1e-5 * uv_err) + para_err * 1e-3
    print(err, para_err)

    return err


def get_parameter_bounds():
    """
    Returns the bounds for the optimization parameters.

    Returns:
    --------
    bounds : list
        A list of tuples, where each tuple contains the lower and upper bounds
        for each parameter to be estimated in the optimization process.

        - angles0: The initial roll, pitch, and yaw angles for the palm.
        - angles1-angles17: Joint angles for the fingers.
        - beta2-beta19: Flexion angles for the finger joints.
        - theta1-theta17: Rotation angles for each finger.
        - q0: The translation (position) of the palm in the 3D space.
    """
    bounds = []
    # Bounds for the initial angles of the palm center (angles0: roll, pitch, yaw)
    bounds.extend([(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])  # angles0 = x[:3]
    # Bounds for the first finger (thumb) angles (angles1: roll, pitch)
    bounds.extend([(0, np.pi), (-np.pi, np.pi)])  # angles1 = x[3:6]
    bounds.extend([(0, np.pi / 2), (-np.pi / 4, np.pi / 4)])  # angles5 = x[6:9]
    bounds.extend([(0, np.pi / 2), (-np.pi / 6, np.pi / 6)])  # angles9 = x[9:12]
    # Bounds for the fourth finger (ring) angles (angles13: roll, pitch)
    bounds.extend([(0, np.pi / 2), (-np.pi / 6, np.pi / 6)])  # angles13 = x[12:15]
    # Bounds for the fifth finger (pinky) angles (ngles17: roll, pitch)
    bounds.extend([(0, np.pi / 2), (-np.pi / 4, np.pi / 4)])  # angles17 = x[15:18]

    # Flexion angles for the finger joints (beta2-beta19: flexion angles)
    bounds.extend([(0, np.pi / 2), (0, np.pi / 2)])  # beta2, beta3 = x[18:20]
    bounds.extend([(0, np.pi / 2), (0, np.pi / 2)])  # beta6, beta7 = x[20:22]
    bounds.extend([(0, np.pi / 2), (0, np.pi / 2)])  # beta10, beta11 = x[22:24]
    bounds.extend([(0, np.pi / 2), (0, np.pi / 2)])  # beta14, beta15 = x[24:26]
    bounds.extend([(0, np.pi / 2), (0, np.pi / 2)])  # beta18, beta19 = x[26:28]
    #
    # bounds.extend([(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)])# theta1, theta5, theta9, theta13, theta17 = x[28: 33]

    # bounds.extend([(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])# theta1, theta5, theta9, theta13, theta17 = x[28: 33]

    # Rotation angles for the joints of each finger (theta1-theta17)
    bounds.extend([(-np.pi, 0), (-np.pi / 6, 0), (-np.pi / 10, 0), (-np.pi / 18, np.pi / 18),
                   (0, np.pi / 6)])  # theta1, theta5, theta9, theta13, theta17 = x[28: 33]
    # Position (translation) of the palm in the 3D space (q0)
    bounds.extend([(-1, 1), (-1, 1), (0, 4)])  # q0 = x[33:36]
    return bounds


def get_init_parameters(p, random=True):
    """
    Initializes the parameters for the hand model.

    Parameters:
    -----------
    p : np.array
        The observed 3D points of the hand landmarks.

    random : bool, optional (default=False)
        If True, the parameters are initialized randomly based on the estimated hand orientation.
        If False, predefined initial parameters are used.

    Returns:
    --------
    x : np.array
        The initialized parameter vector with angles and translation values.
        The vector contains:
        - angles0: Initial palm angles (roll, pitch, yaw)
        - angles1-angles17: Initial joint angles for each finger
        - q0: Initial position of the palm in 3D space
    """
    if random:
        # Random initialization based on estimating hand orientation from positions
        T_c_0 = estimate_hand_orientation(p,True)
        R = sR.from_matrix(T_c_0[:3, :3])  # Convert the rotation matrix to Euler angles
        angles = R.as_euler('xyz', degrees=False)  # Convert rotation matrix to Euler angles
        x = np.zeros(31)
        x[:3] = angles  # Set the initial angles of the palm center
        x[23:28] = np.array([-8.43884794e-01,-2.98872942e-01, -5.73973535e-02,  1.67494017e-01,  3.96680375e-01])
        x[28:] = T_c_0[:3, 3]  # Set the position of the palm center
    else:
        x = np.array([-3.12168364e+00, -1.48777228e-01, -1.82606943e-01,
                      -8.67605941e-02, 1.37455854e-01,
                      5.20622723e-01, -2.04708417e-01,
                      -3.03428774e+00, -2.85623270e-01,
                      2.70712313e+00, -2.91955177e-01,
                      2.42456372e+00, -2.72234764e-01,
                      1.61641834e-01, 5.43853132e-02,
                      6.41770029e-05, 3.58922991e-05,
                      1.44549156e-01, 1.56333097e-02,
                      1.68016999e-01, 4.80753593e-02,
                      1.01163674e-01, 1.14735612e-01,
                      1.29742734e+00, 4.94257173e-01, 2.82316014e-01, 6.85396618e-02, 4.59913656e-05,
                      -9.19271204e-02, 5.49594579e-02, 3.65995441e-01])
    return x


def get_axis_from(rotation, point, scale=0.01):
    """
    Computes the axis vectors for a given rotation and point in 3D space.

    Parameters:
    -----------
    rotation : np.array
        A 3x3 rotation matrix representing the orientation of the local frame.

    point : np.array
        A 3D point representing the origin of the local frame in space.

    scale : float, optional (default=0.01)
        A scaling factor to adjust the length of the axis vectors for visualization.

    Returns:
    --------
    axis : np.array
        A 3x3x2 array containing the coordinates of the axis vectors in 3D space.
        - axis[i, :, 0] is the origin of the i-th axis (i = 0,1,2 -> X, Y, Z).
        - axis[i, :, 1] is the end point of the i-th axis after applying the rotation.
    """
    # Scale the rotation matrix for visualization purposes
    rotation = rotation * scale
    # Initialize a 3x3x2 array to store axis vectors (X, Y, Z)
    axis = np.zeros([3, 3, 2])
    # X-axis
    axis[0, :, 0] = point[:3]  # Origin of the axis
    axis[0, :, 1] = point[:3] + rotation[:3, 0]  # End point of the X-axis

    axis[1, :, 0] = point[:3]  # Origin of the axis
    axis[1, :, 1] = point[:3] + rotation[:3, 1]  # End point of the Y-axis

    axis[2, :, 0] = point[:3]  # Origin of the axis
    axis[2, :, 1] = point[:3] + rotation[:3, 2]  # End point of the Z-axis

    return axis


