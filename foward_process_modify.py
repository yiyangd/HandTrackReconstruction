import os
import pdb

import cv2
import numpy as np
from process_video2unity import plot_global_frame
from fit_hand_model_modify import get_axis_from, fit_hand_model, foward_hand_model, get_init_parameters, get_parameter_bounds, minimize, print_pred_results
import matplotlib.pyplot as plt
from glob import glob
from hand_landmark_estimation import estimate_hand_lengths, get_initial_hand_lengths

def visualize_landmarks_3d(p_landmarks, global_frames, show_frame=False):
    """
    Visualizes the hand landmarks in 3D.

    Args:
        landmarks: List of landmarks detected by MediaPipe.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define connections between landmarks to visualize the hand structure
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    # Extract x, y, z coordinates
    xs = p_landmarks[:, 0]
    ys = p_landmarks[:, 1]
    zs = p_landmarks[:, 2]

    # Plot each landmark
    ax.scatter(xs, ys, zs, c='k', marker='o', s=50)

    # Plot connections
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], [zs[start_idx], zs[end_idx]], 'k', linewidth=2)
        plot_global_frame(ax, np.eye(4))
    if show_frame:
        plot_global_frame(ax, global_frames['T_c_0'])
        plot_global_frame(ax, global_frames['T_c_0'])
        plot_global_frame(ax, global_frames['T_c_1'])
        plot_global_frame(ax, global_frames['T_c_2'])
        plot_global_frame(ax, global_frames['T_c_3'])

        plot_global_frame(ax, global_frames['T_c_5'])
        plot_global_frame(ax, global_frames['T_c_6'])
        plot_global_frame(ax, global_frames['T_c_7'])

        plot_global_frame(ax, global_frames['T_c_9'])
        plot_global_frame(ax, global_frames['T_c_10'])
        plot_global_frame(ax, global_frames['T_c_11'])

        plot_global_frame(ax, global_frames['T_c_13'])
        plot_global_frame(ax, global_frames['T_c_14'])
        plot_global_frame(ax, global_frames['T_c_15'])

        plot_global_frame(ax, global_frames['T_c_17'])
        plot_global_frame(ax, global_frames['T_c_18'])
        plot_global_frame(ax, global_frames['T_c_19'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

lengths = np.array([[0.03997489, 0.02405818, 0.01993143, 0.015], # 大拇指两个关节之间的距离
 [0.0898971,  0.035091,   0.0204093,  0.017],# 食指
 [0.08721813, 0.03997809, 0.02406639, 0.01702162],#中指
 [0.08523586, 0.03506496, 0.02118049, 0.01692583], # 无名指
 [0.08316337, 0.02717291, 0.01591242, 0.01818702]]) # 小拇指

pred_x = np.array(
    [-2.655, -2.799,  2.971, # 手掌的局部坐标系朝向矩阵
     -7.14828592e-02, - 5.40555167e-01, # 大拇指第一关节局部坐标系
     5.08021775e-01, - 3.05094508e-02, # 食指第一关节局部坐标系
     3.05737921e-01, - 5.78265211e-02, # 中指第一关节局部坐标系
     2.65561354e-01, - 2.17104682e-02, # 无名指第一关节局部坐标系
     1.69063853e-01, 5.07512197e-02, # 小拇指第一关节局部坐标系
     1.22232308e-04+0 * np.pi / 180,  3.14550955e-02+0 * np.pi / 180, #大拇指 2和3关节的旋转角
     6.11638852e-02+0 * np.pi / 180, 1.40305016e-01+ 0 * np.pi / 180, # 食指 2和3关节的旋转角
     3.92698071e-01+0 * np.pi / 180,  6.41770032e-05+0 * np.pi / 180, # 中指 2和3关节的旋转角
     3.66759591e-02+0 * np.pi / 180,  2.92101027e-01+0 * np.pi / 180, # 无名指 2和3关节的旋转角
     1.92225491e-04+0 * np.pi / 180,  6.41770029e-05+0 * np.pi / 180, # 小拇指 2和3关节的旋转角
     -1.21904504e+00, - 3.77847047e-01, - 8.78549544e-02,  1.44105227e-01,  3.74252200e-01,
     .0, .0,  0.1] # 手掌在相机坐标系下的位置
)

pred_uv, pred_p, global_frames = foward_hand_model(pred_x, lengths)
visualize_landmarks_3d(pred_p, global_frames, show_frame=False)