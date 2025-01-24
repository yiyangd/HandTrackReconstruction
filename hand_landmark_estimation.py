import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sR
# from fit_hand_model_modify import local_frame_to_list

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
    R_c_0 = Angle2RotationMatrix([np.pi, 0, 0]) @ R_c_0
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
def get_global_rotation_matrices(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10, T_10_11, T_13_14, T_14_15, T_17_18, T_18_19, T_3_4=np.eye(4), T_7_8=np.eye(4), T_11_12=np.eye(4), T_15_16=np.eye(4), T_19_20=np.eye(4)):
    T_c_1 = T_c_0 @ T_0_1
    T_c_2 = T_c_1 @ T_1_2
    T_c_3 = T_c_2 @ T_2_3
    T_c_4 = T_c_3 @ T_3_4

    T_c_5 = T_c_0 @ T_0_5
    T_c_6 = T_c_5 @ T_5_6
    T_c_7 = T_c_6 @ T_6_7
    T_c_8 = T_c_7 @ T_7_8

    T_c_9 = T_c_0 @ T_0_9
    T_c_10 = T_c_9 @ T_9_10
    T_c_11 = T_c_10 @ T_10_11
    T_c_12 = T_c_11 @ T_11_12

    T_c_13 = T_c_0 @ T_0_13
    T_c_14 = T_c_13 @ T_13_14
    T_c_15 = T_c_14 @ T_14_15
    T_c_16 = T_c_15 @ T_15_16

    T_c_17 = T_c_0 @ T_0_17
    T_c_18 = T_c_17 @ T_17_18
    T_c_19 = T_c_18 @ T_18_19
    T_c_20 = T_c_19 @ T_19_20
    pred_p = [T_c_0[:3, 3], T_c_1[:3, 3],T_c_2[:3, 3],T_c_3[:3, 3],T_c_4[:3, 3],T_c_5[:3, 3],T_c_6[:3, 3],T_c_7[:3, 3],T_c_8[:3, 3],T_c_9[:3, 3],T_c_10[:3, 3],T_c_11[:3, 3],T_c_12[:3, 3],T_c_13[:3, 3],T_c_14[:3, 3],T_c_15[:3, 3],T_c_16[:3, 3],T_c_17[:3, 3],T_c_18[:3, 3],T_c_19[:3, 3],T_c_20[:3, 3]]
    pred_p = np.stack(pred_p, 0)

    global_frames = {'T_c_0': T_c_0,'T_c_1': T_c_1, 'T_c_2':T_c_2,'T_c_3':T_c_3,
                        'T_c_5':T_c_5, 'T_c_6':T_c_6,'T_c_7':T_c_7, 'T_c_9':T_c_9,
                        'T_c_10':T_c_10,'T_c_11':T_c_11, 'T_c_13':T_c_13, 'T_c_14':T_c_14,
                        'T_c_15':T_c_15, 'T_c_17':T_c_17, 'T_c_18':T_c_18,'T_c_19':T_c_19}
    return global_frames, pred_p

def estimate_plane(points, point_out=False):
     
    # 将点云数据转换为矩阵A和向量b
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = points[:, 2]
    
    # 使用最小二乘法计算平面方程系数
    coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 平面方程的系数
    a, b, c = coefficients
    
    # 计算平面法向量
    normal_vector = np.array([a, b, -1])
    if point_out:
        normal_vector = -normal_vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 归一化法向量
    
    return normal_vector



def estimate_hand_orientation(points, point_out=False):
    # p0 = landmarks[0]
    # p5 = landmarks[5]
    # p17 = landmarks[17]
    # points = landmarks[[0, 5, 17]]
    palm_z = estimate_plane(points[[0, 5, 9, 13]], point_out)
    palm_y = points[13] - points[5]
    palm_y = palm_y / np.linalg.norm(palm_y)
    palm_x = np.cross(palm_y, palm_z)
    palm_y = np.cross(palm_z, palm_x)
    
    R = np.stack([palm_x, palm_y, palm_z], -1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = points[0]
    return T
def calculate_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1*norm2 == 0:
        return 0
    cos_theta = dot_product / (norm1 * norm2)
    
    angle_radians = np.arccos(cos_theta)

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
def Angle2RotationMatrix(angles):
    r, p, y = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R
def compute_coordinates_in_wrist(points, T0):
    inverse_T = np.linalg.inv(T0)
    points_0 = points @ inverse_T[:3, :3].T + inverse_T[:3, 3:4].T
    points_0[[1, 5, 9, 13, 17], 2] = 0
    return points_0
def compute_offset_angles(points0):
    root_landmarks = points0[[1, 5, 9, 13, 17]]
    root_landmarks[:, 2] = 0
    x_axis = np.array([0, 1, 0])
    theta_1 = 90 - calculate_angle(x_axis, root_landmarks[0])
    theta_5 = 90 - calculate_angle(x_axis, root_landmarks[1])
    theta_9 = 90 - calculate_angle(x_axis, root_landmarks[2])
    theta_13 = 90 - calculate_angle(x_axis, root_landmarks[3])
    theta_17 = 90 - calculate_angle(x_axis, root_landmarks[4])
    return theta_1, theta_5, theta_9, theta_13, theta_17
def compute_single_root_local_frame(points0):
    x_axis = points0[1] - points0[0]
    x_axis = x_axis / np.linalg.norm(x_axis, 2, -1)
    tmp_z_axis = np.array([0, 0, 1])
    y_axis = np.cross(tmp_z_axis, x_axis)
    if np.linalg.norm(y_axis, 2, -1) != 0:
        y_axis = y_axis / np.linalg.norm(y_axis, 2, -1)
    else:
        y_axis = [0,1,0]
    if y_axis[1] < 0:
        y_axis = -y_axis
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], -1)
    tmp_R = sR.from_matrix(R)
    angles = tmp_R.as_euler('xyz', degrees=False)
    beta, gamma = angles[1:3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = points0[0]

    # gamma = calculate_angle(y_axis, np.array([0, 1, 0]))
    # gamma = -gamma if x_axis[1] < 0 else gamma
    # project_axis = np.array([x_axis[0], x_axis[1], 0])
    # beta = calculate_angle(x_axis, project_axis)
    # beta = - beta if x_axis[2] > 0 else beta
    # print(calculate_angle(project_axis, np.array([1, 0, 0])) * np.pi / 180)
    # print(Angle2RotationMatrix([0, beta, gamma]) - R)
    # print(Angle2RotationMatrix([0, -beta, gamma]) - R)
    # print(Angle2RotationMatrix([0, beta, -gamma]) - R)
    # print(Angle2RotationMatrix([0, -beta, -gamma]) - R)

    return T, beta, gamma

def compute_root_joint_local_frames(points0):
    T0_1, beta1, gamma1 = compute_single_root_local_frame(points0[[1, 2]])
    T0_5, beta5, gamma5 = compute_single_root_local_frame(points0[[5, 6]])
    T0_9, beta9, gamma9 = compute_single_root_local_frame(points0[[9, 10]])
    T0_13, beta13, gamma13 = compute_single_root_local_frame(points0[[13, 14]])
    T0_17, beta17, gamma17 = compute_single_root_local_frame(points0[[17, 18]])
    return T0_1, T0_5, T0_9, T0_13, T0_17, beta1, beta5, beta9, beta13, beta17, gamma1, gamma5, gamma9, gamma13, gamma17
def compute_middle_joint_local_frame(points0, T_0_i):
    inverse_T = np.linalg.inv(T_0_i)
    points0 = points0 @ inverse_T[:3, :3].T + inverse_T[:3, 3:4].T
    points0[:, 1] = 0
    vector12, vector23, vector34 = points0[1] - points0[0], points0[2] - points0[1], points0[3] - points0[2]
    beta1, beta2 = calculate_angle(vector12, vector23), calculate_angle(vector23, vector34)
    if points0[2, 2] > points0[1, 2]:
        beta1 = -beta1
    if points0[3, 2] > points0[2, 2]:
        beta2 = -beta2
    # may be need to judge neg or pos of angles
    R1 = Angle2RotationMatrix([0, beta1 * np.pi / 180, 0])
    R2 = Angle2RotationMatrix([0, beta2 * np.pi / 180, 0])
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[0, 3] = np.linalg.norm(vector12, 2, -1)

    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[0, 3] = np.linalg.norm(vector23, 2, -1)

    T3 = np.eye(4)
    T3[0, 3] = np.linalg.norm(vector34, 2, -1)

    return T1, T2, T3, beta1, beta2
def point2pixel(points):
    points = points.reshape(-1, 3)
    intrinsics = np.array([[609.62, 0,  323.44],
                           [0, 609.67, 247.87],
                           [0, 0, 1]])
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]

    u = np.round(x * intrinsics[0, 0] + intrinsics[0, 2])
    v = np.round(y * intrinsics[1, 1] + intrinsics[1, 2])
    pixel = np.stack([u, v], -1)
    return pixel

def compute_local_frames_by_pts(points, uv, outliers):
    # compute palm orientation and location
    #import pdb; pdb.set_trace()
    T_c_0 = estimate_hand_orientation(points, True)
    R_c_0 = sR.from_matrix(T_c_0[:3, :3])
    angles = R_c_0.as_euler('xyz', degrees=False)
    alpha_0, beta_0, gamma_0 = angles

    points0 = compute_coordinates_in_wrist(points, T_c_0)
    theta_1, theta_5, theta_9, theta_13, theta_17 = compute_offset_angles(points0)
    T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, beta1, beta5, beta9, beta13, beta17, gamma1, gamma5, gamma9, gamma13, gamma17 = compute_root_joint_local_frames(points0)

    T_1_2, T_2_3, T_3_4, beta2, beta3 = compute_middle_joint_local_frame(points0[[1, 2, 3, 4]], T_0_1)
    T_5_6, T_6_7, T_7_8, beta6, beta7 = compute_middle_joint_local_frame(points0[[5, 6, 7, 8]], T_0_5)
    T_9_10, T_10_11, T_11_12, beta10, beta11 = compute_middle_joint_local_frame(points0[[9, 10, 11, 12]], T_0_9)
    T_13_14, T_14_15, T_15_16, beta14, beta15 = compute_middle_joint_local_frame(points0[[13, 14, 15, 16]], T_0_13)
    T_17_18, T_18_19, T_19_20, beta18, beta19 = compute_middle_joint_local_frame(points0[[17, 18, 19, 20]], T_0_17)

    global_frames, pred_p = get_global_rotation_matrices(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
                                     T_10_11, T_13_14, T_14_15, T_17_18, T_18_19, T_3_4, T_7_8, T_11_12, T_15_16, T_19_20)
    pred_x = np.array([alpha_0, beta_0, gamma_0, beta1, gamma1, beta5, gamma5, beta9, gamma9, beta13, gamma13, beta17, gamma17,
                       beta2, beta3, beta6, beta7, beta10, beta11, beta14, beta15, beta18, beta19, theta_1, theta_5, theta_9, theta_13, theta_17, pred_p[0, 0], pred_p[0, 1], pred_p[0, 2]])
    pred_uv = point2pixel(pred_p)

    uv_err = np.sqrt(np.sum((pred_uv - uv) ** 2, -1))
    p_err = np.sqrt(np.sum((pred_p[:, :3] - points) ** 2, -1))
    err = np.sum(p_err * (1 - outliers) + 1e-5 * uv_err)

    #import pdb; pdb.set_trace()
    data_list = local_frame_to_list(T_c_0, T_0_1, T_0_5, T_0_9, T_0_13, T_0_17, T_1_2, T_2_3, T_5_6, T_6_7, T_9_10,
                                    T_10_11, T_13_14, T_14_15, T_17_18, T_18_19)

    return pred_x, pred_p, global_frames, outliers, 0, err, data_list







def get_initial_hand_lengths():  # in cm
    current_lengths = np.zeros([5,4])
    current_lengths[0, 0] = 4
    current_lengths[0, 1] = 3.4
    current_lengths[0, 2] = 3
    current_lengths[0, 3] = 2.5

    current_lengths[1, 0] = 9
    current_lengths[1, 1] = 3.5
    current_lengths[1, 2] = 2
    current_lengths[1, 3] = 1.7

    current_lengths[2, 0] = 8.7
    current_lengths[2, 1] = 4
    current_lengths[2, 2] = 2.4
    current_lengths[2, 3] = 1.7

    current_lengths[3, 0] = 8.5
    current_lengths[3, 1] = 3.5
    current_lengths[3, 2] = 2.1
    current_lengths[3, 3] = 1.7

    current_lengths[4, 0] = 8.3
    current_lengths[4, 1] = 2.7
    current_lengths[4, 2] = 1.6
    current_lengths[4, 3] = 1.8

    current_lengths *= 1e-2

    min_lengths = current_lengths * (1 - 0.3)
    max_lengths = current_lengths * (1 + 0.3)
   

    return current_lengths, min_lengths, max_lengths



def estimate_hand_lengths(points, lengths, min_lengths, max_lengths, distance_thresh=0.05, delta=0.9, use_current_length=False):
    """
    input:
    points: 21*3, current xyz of 21 landmarks
    lengths: 5*4, 20 lengths between landmarks, 0-1, 1-2, 2-3, 3-4; ...; 0-17, 17-18, 18-19, 19-20
    distance_thresh: check if the input points are reasonable
    output:
    updated_lengths: 5*4, 20 lengths between landmarks, 0-1, 1-2, 2-3, 3-4; ...; 0-17, 17-18, 18-19, 19-20
    """

    current_lengths = np.zeros([5,4])
    current_lengths[0, 0] = np.linalg.norm(points[1] - points[0], 2, -1) 
    current_lengths[0, 1] = np.linalg.norm(points[2] - points[1], 2, -1) 
    current_lengths[0, 2] = np.linalg.norm(points[3] - points[2], 2, -1) 
    current_lengths[0, 3] = np.linalg.norm(points[4] - points[3], 2, -1) 

    current_lengths[1, 0] = np.linalg.norm(points[5] - points[0], 2, -1) 
    current_lengths[1, 1] = np.linalg.norm(points[6] - points[5], 2, -1) 
    current_lengths[1, 2] = np.linalg.norm(points[7] - points[6], 2, -1) 
    current_lengths[1, 3] = np.linalg.norm(points[8] - points[7], 2, -1) 

    current_lengths[2, 0] = np.linalg.norm(points[9] - points[0], 2, -1) 
    current_lengths[2, 1] = np.linalg.norm(points[10] - points[9], 2, -1) 
    current_lengths[2, 2] = np.linalg.norm(points[11] - points[10], 2, -1) 
    current_lengths[2, 3] = np.linalg.norm(points[12] - points[11], 2, -1) 

    current_lengths[3, 0] = np.linalg.norm(points[13] - points[0], 2, -1) 
    current_lengths[3, 1] = np.linalg.norm(points[14] - points[13], 2, -1) 
    current_lengths[3, 2] = np.linalg.norm(points[15] - points[14], 2, -1) 
    current_lengths[3, 3] = np.linalg.norm(points[16] - points[15], 2, -1) 

    current_lengths[4, 0] = np.linalg.norm(points[17] - points[0], 2, -1) 
    current_lengths[4, 1] = np.linalg.norm(points[18] - points[17], 2, -1) 
    current_lengths[4, 2] = np.linalg.norm(points[19] - points[18], 2, -1) 
    current_lengths[4, 3] = np.linalg.norm(points[20] - points[19], 2, -1) 

    if use_current_length:
        min_lengths = current_lengths * (1 - 0.3)
        max_lengths = current_lengths * (1 + 0.3)
        updated_lengths = current_lengths
    else:
        error = lengths - current_lengths
        outliers = np.abs(error) > distance_thresh
        # if outlier: using history lengths, else: using current_lengths * delta + lengths * (1 - delta), eg: current_length: 1.1cm, length: 1.2cm, delta: 0.1, 1.1*0.1+1.2*0.9 = 1.19
        updated_lengths = lengths * outliers + (1 - outliers) * (current_lengths * delta + lengths * (1 - delta))

        updated_lengths = np.minimum(updated_lengths, max_lengths)
        updated_lengths = np.maximum(updated_lengths, min_lengths)

    return updated_lengths, min_lengths, max_lengths

def draw_finger_lengthts(measurement, estimation):
    measurement = np.stack(measurement, 0)
    estimation = np.stack(estimation, 0)
    # Create a figure with 5x4 subplots
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))

    # Loop to create and plot data in each subplot
    for i in range(5):
        for j in range(4):
            y1, y2 = measurement[:, i, j], estimation[:, i, j]
            ax = axes[i, j]
            ax.plot(y1, label='measurement')
            ax.plot(y2, label='estimation')
            ax.set_title(f'Subplot {i*4+j+1}')
            ax.legend()

    plt.tight_layout()
    plt.show()



def main():
    # 示例点云数据
    # points = np.array( [
    # [-6.25, -2.51, 33.9],
    # [1.79, 2.01, 31.50],
    # [1.70, -3.40, 32.8],
    # ])

    # normal_vector = estimate_hand_orientation(points)

    print(np.linalg.norm([1, 1, 1], 2,   -1))

main()