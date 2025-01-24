import os
import pdb
import time
import cv2
import numpy as np
import socket
# Import functions for fitting and manipulating the hand model
from fit_hand_model_modify import get_axis_from, fit_hand_model, foward_hand_model, get_init_parameters, \
    get_parameter_bounds, minimize, print_pred_results
import matplotlib.pyplot as plt
from glob import glob
from hand_landmark_estimation import estimate_hand_lengths, get_initial_hand_lengths, estimate_hand_orientation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Function to plot the global coordinate frame
def plot_global_frame(ax, T, length=0.02):
    """
    Plots the global coordinate frame using quiver arrows.

    Args:
        ax: Matplotlib 3D axis.
        T: Transformation matrix (rotation and translation).
        length: Length of the arrows representing the coordinate axes.
    """
    R = T[:3, :3].T  # Extract rotation matrix
    p = T[:3, 3]  # Extract translation vector
    # Plot quiver for each axis (x: red, y: green, z: blue)
    for i, color in enumerate(['r', 'g', 'b']):
        ax.quiver(p[0], p[1], p[2], R[i, 0], R[i, 1], R[i, 2], color=color, length=length)


# Function to visualize the 3D hand landmarks
def visualize_landmarks_3d(m_landmarks, p_landmarks, image, global_frames, outliers):
    """
    Visualizes the hand landmarks in 3D.

    Args:
        m_landmarks: Modeled landmarks.
        p_landmarks: Predicted landmarks.
        image: Input image for visualization.
        global_frames: Dictionary of global frames for different parts of the hand.
        outliers: Array of outliers detected in the hand landmarks.
    """
    print("Measured 3D Position:\n")
    print(m_landmarks)
    fig = plt.figure()
    ax0 = fig.add_subplot(121)  # 2D plot of the image
    ax0.imshow(image)
    ax = fig.add_subplot(122, projection='3d')  # 3D plot for landmarks

    # Define connections between landmarks to visualize the hand structure
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]
    indices = [0, 1, 5, 6, 9, 13, 17]
    # Estimated Colorful skeleton
    # Extract x, y, z coordinates
    if True:
        
        if p_landmarks is not None:
            print("Estimated 3D Position:\n")
            print(p_landmarks)
            xp = p_landmarks[:, 0]
            yp = p_landmarks[:, 1]
            zp = p_landmarks[:, 2]

            # Plot each landmark
            # ax.scatter(xp[indices], yp[indices], zp[indices], c='m', marker='*', s=50)
            ax.scatter(xp, yp, zp,  c='k', marker='o')

            # Plot connections
            for connection in connections:
                start_idx, end_idx = connection
                # ax.plot([xp[start_idx], xp[end_idx]], [yp[start_idx], yp[end_idx]], [zp[start_idx], zp[end_idx]], 'm')
                ax.plot([xp[start_idx], xp[end_idx]], [yp[start_idx], yp[end_idx]], [zp[start_idx], zp[end_idx]], 'k')

    # Measured Black Skeleton
    # Extract x, y, z coordinates of modeled landmarks
    # m_landmarks = m_landmarks[outliers < 0.5]
    xs = m_landmarks[:, 0]
    ys = m_landmarks[:, 1]
    zs = m_landmarks[:, 2]
    # Indices to select
    

    
    # Plot modeled landmarks and their connections
    # ax.scatter(xs[indices], ys[indices], zs[indices], c='k', marker='o')
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], [zs[start_idx], zs[end_idx]], 'k')

    # Plot global frames for different parts of the hand (e.g., wrist, fingers)
    if False:
        # for i in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
        for i in [5,6,7]:
            plot_global_frame(ax, global_frames[f'T_c_{i}'])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_tick_params(pad=20)  # 调整X轴标签与刻度的距离
    ax.yaxis.set_tick_params(pad=20)  # 调整Y轴标签与刻度的距离
    #ax.yaxis.set_ticklabels([])  
    ax.zaxis.set_tick_params(pad=20)  # 调整Z轴标签与刻度的距离
    ax.set_xlabel('X', fontsize=15, labelpad=35)
    ax.set_ylabel('Y', fontsize=15, labelpad=35)
    ax.set_zlabel('Z', fontsize=15, labelpad=35)
    ax.set_zlim(min(zs), max(zs))
    ax.view_init(elev=-81, azim=-90.01)
    ax.dist = 2
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    #import pdb; pdb.set_trace()


# Function to compute uncertainty of hand landmark measurements using covariance
def compute_uncertainty_matrix(pts):
    """
    Computes the uncertainty based on covariance matrix of the points.

    Args:
        pts: Array of points representing the landmarks.

    Returns:
        Uncertainty value based on eigenvalues of the covariance matrix.
    """
    conv = np.cov(pts.T)
    eig_values, eig_vectors = np.linalg.eig(conv)
    eig_values.sort()  # Sort eigenvalues
    if eig_values[1] < 1e-2:
        return 0.  # Low uncertainty if eigenvalues are small
    elif eig_values[0] < 1e-2:
        return 0.5
    else:
        return 0


# Function to detect outliers in the landmark positions
def outlier_detection(p):
    """
    Detects outliers in hand landmark points.

    Args:
        p: Hand landmark positions.

    Returns:
        outliers0: Array of detected outliers (0 or 1).
    """
    # Compute median of points
    median_xyz = np.median(p, 0)
    # Distance from median
    distances = np.sqrt(np.sum((p - median_xyz[None]) ** 2, -1))
    # Detect large deviations as outliers
    outliers0 = (distances > 0.2).astype(np.float32)

    # Find non-obvious false measurements
    # Further refine outliers based on uncertainties for different hand parts
    confidence1 = compute_uncertainty_matrix(p[1:5])  # 1-4
    confidence5 = compute_uncertainty_matrix(p[5:9])  # 1-4
    confidence9 = compute_uncertainty_matrix(p[9:13])  # 1-4
    confidence13 = compute_uncertainty_matrix(p[13:17])  # 1-4
    confidence17 = compute_uncertainty_matrix(p[17:21])  # 1-4
    outliers0[1:5] += confidence1
    outliers0[5:9] += confidence5
    outliers0[9:13] += confidence9
    outliers0[13:17] += confidence13
    outliers0[17:21] += confidence17
    # for start_idx in [1, 5, 9, 13, 17]:
    #     confidence = compute_uncertainty_matrix(p[start_idx:start_idx + 4])
    #     outliers0[start_idx:start_idx + 4] += confidence

    outliers0 = np.clip(outliers0, a_min=0, a_max=1.)  # Clip values to [0, 1]

    return outliers0
def compute_point_prob(p_pred, p_measure, sigma=0.02):
    probs = []
    for i in range(p_pred.shape[0]):
        dist = np.linalg.norm(p_pred[i, :3] - p_measure[i], 2, -1)
        prob = np.exp(- dist**2 / (2 * sigma**2)) / np.sqrt(2*np.pi*sigma**2)
        probs.append(prob)
    print(probs / np.max(probs))
    return np.array(probs)
def visualize_optimization_process(m_landmarks, image, lengths, uv, outliers):
    global history
    iter_n = len(history)
    print(iter_n)
    step = 1
    loss = []
    for i in range(0, iter_n):
        pred_x = history[i]
        iter_loss = fit_hand_model(pred_x, m_landmarks, uv, lengths, outliers)
        loss.append(iter_loss)
    x = list(range(0, iter_n))

    # 绘制主折线图
    fig, ax = plt.subplots()
    ax.plot(x, loss, marker='o')

    # 添加标题和标签
    ax.set_xticks(range(int(x[5]), int(x[-1]) + 1,5))  # 确保刻度是整数范围
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.xaxis.set_tick_params(pad=15)  # 调整X轴标签与刻度的距离
    ax.yaxis.set_tick_params(pad=15)  # 调整Y轴标签与刻度的距离
    ax.set_xlabel('Iteration', fontsize=35, labelpad=25)
    ax.set_ylabel('Loss', fontsize=35, labelpad=25)
    ax.grid()

    # 创建局部放大图
    ax_inset = inset_axes(ax, width="60%", height="60%", loc='upper right')

    # 画出后10个数的折线图
    ax_inset.plot(x[5:], loss[5:], marker='o', color='red')
    #ax_inset.set_xticks(x[1:])  # 设置x轴刻度
    ax_inset.set_xticks(range(int(x[5]), int(x[-1]) + 1,5))  # 确保刻度是整数范围
    ax_inset.tick_params(axis='both', which='major', labelsize=25)
    ax_inset.set_xlabel('Iteration', fontsize=30, labelpad=25)
    ax_inset.set_ylabel('Loss', fontsize=30, labelpad=25)

    # ax.annotate('',
    #             xy=(18, loss[18]),  # 箭头指向放大图的位置
    #             xytext=(15, loss[15]),  # 箭头起点
    #             arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 显示图形
    plt.show()
    
    for i in range(0, iter_n, step):
        pred_x = history[i]
        pred_uv, pred_p, global_frames, _ = foward_hand_model(pred_x, lengths)
        probs = compute_point_prob(pred_p, m_landmarks)
        print("iter " + str(i) + " over " + str(iter_n) + " iterations")
        print(probs)
        visualize_landmarks_3d(m_landmarks, pred_p, image,global_frames, outliers = None)
def callback(x):
    global history
    history.append(x)

# Function to process a single frame of hand tracking
def process_single_frame(p, uv, x, lengths, img=None):
    """
    Processes a single frame of hand landmark tracking.

    Args:
        p: 3D coordinates of hand landmarks.
        uv: 2D coordinates of hand landmarks.
        x: Model parameters for the hand.
        lengths: Finger lengths.

    Returns:
        Optimized parameters, predicted positions, global frames, and outliers.
    """
    outliers = outlier_detection(p)  # Detect outliers

    # If initial parameters not provided, generate them randomly
    if x is None:
        x = get_init_parameters(p, random=True)

    # Set base position of hand model
    x[28:] = p[0]
    strat_a_time = time.time()
    global history
    history = [x]

    # Optimize the hand model parameters using Powell method
    result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths, outliers), bounds=get_parameter_bounds(),
                      method="Powell", options={'maxiter': 50, 'gtol': 1e-1, 'eps':1}, callback=callback)
    a_time = time.time() - strat_a_time
    visualize_optimization_process(p, img, lengths, uv, outliers)
    print(f"Time taken for minimization: {a_time:.4f} seconds")
    # import pdb; pdb.set_trace()
    # gtol 梯度范数上限，norm 范数浮动上限， eps 求解浮动上限
    # result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths),
    #                   bounds=get_parameter_bounds(), method="bfgs",
    #                   options={'gtol': 1e-3, 'eps': 1e-3})

    # result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths, outliers),
    #                   bounds=get_parameter_bounds(), method="nelder-mead",
    #                   options={'gtol': 1e-9, 'eps': 1e-9})

    print(f'Global Minimum: {result.fun}')  # Print the final error of the optimization
    # print(f'x = {result.x}')
    # Use the optimized parameters to predict the hand pose
    pred_x = result.x  # Get the predicted model parameters
    # Predict 3D points and frames
    pred_uv, pred_p, global_frames, data_list = foward_hand_model(pred_x, lengths)


    return pred_x, pred_p[:, :3], global_frames, outliers, data_list


# Function to process a video sequence of hand motions and fit the hand model to each frame
def process_single_video(data_path, load_pre_save_data=False, visualize=True, save_results=False):
    """
    Processes all frames in a video to track hand landmarks.

    Args:
        data_path: Path to the directory containing video frames and data.
        load_pre_save_data: Whether to load pre-saved prediction data.
        visualize: Whether to visualize the results.
        save_results: Whether to save the results.
    """
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Set the address and port for sending data
    serverAddressPort = ("127.0.0.1", 5052)
    # Get the number of frames
    n_frame = len(glob(os.path.join(data_path, f'annotate_img_*.png')))
    pred_x = None
    # Initialize hand lengths
    current_lengths, min_lengths, max_lengths = get_initial_hand_lengths()
    a = [0,10,35,45,50,55,60]
    #a = [25,40,60,85,130,160,170,175,195,200,205]
    b = [25,40,50,60,70, 130,140, 150, 160,170, 180, 195]
    # Process frames with a step of 5
    for frame_i in range(0, n_frame, 5):
        # Path to the image frame
        # if frame_i  not in  b:
        #     continue
        try:
            img_path = os.path.join(data_path, f'annotate_img_{frame_i}.png')
            # Path to the landmarks data
            kpts_path = os.path.join(data_path, f'data_{frame_i}.txt')
        except FileNotFoundError as e:
            continue
        # Read the image
        img = cv2.imread(img_path)
        # Load the landmark points
        kpts = np.loadtxt(kpts_path).astype(np.float32)
        # Skip frame if landmarks are incomplete
        if kpts.shape[0] != 21:
            continue
        p = kpts[:, 2:]  # 3D points (x, y, z)
        p[:, 1] = -p[:, 1]
        uv = kpts[:, :2]  # 2D image coordinates
        #print(uv)

        # Display the image
        cv2.imshow("Hand", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Stop the video if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        if load_pre_save_data:
            # Load previously saved results
            pred_data = np.load(os.path.join(data_path, f'pred_data1_{frame_i}.txt.npz'), allow_pickle=True)

            pred_x, pred_p, global_frames, outliers, data_list = pred_data['pred_x'], pred_data['pred_kpts'], pred_data[
                'global_rotations'].item(), pred_data['outliers'], pred_data['data_list']
            data_list = data_list.tolist()
        else:
            # Estimate hand lengths and process the frame
            # current_lengths, _ = estimate_hand_lengths(p, current_lengths, min_lengths, max_lengths, distance_thresh=0.02, delta=0.2)
            print("Start to Process Frame " + str(frame_i))
            #import pdb; pdb.set_trace()
            pred_x, pred_p, global_frames, outliers, data_list = process_single_frame(p, uv, pred_x, current_lengths, img)
            # Append predicted root point to data list
            data_list.extend([pred_p[0, 0], -pred_p[0, 1], pred_p[0, 2]])
        # import pdb; pdb.set_trace()
        print(frame_i)
        print("2D Pixel:\n")
        print(uv)
        # Visualize the results
        if visualize:
            #print(global_frames)
            visualize_landmarks_3d(p, pred_p[:, :3], img, global_frames, outliers)
            visualize_landmarks_3d(p, None, img, global_frames, outliers)

        
        # Save results to file
        if save_results:
            np.savez_compressed(os.path.join(data_path, f'pred_data1_{frame_i}.txt'),
                                pred_kpts=pred_p,
                                pred_x=pred_x,
                                global_rotations=global_frames,
                                outliers=outliers,
                                data_list=np.array(data_list))
        # Send data via UDP
        print(data_list)
        sock.sendto(str.encode(str(data_list)), serverAddressPort)
        # import pdb; pdb.set_trace()
        # Delay to simulate real-time processing
        time.sleep(1)

    # Release resources and stop the pipeline
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # List of experiment folders
    folders1 = ['grasp']
    folders2 = ['prediction_70']
    # folders = ['prediction_71','rotate_z','finger-rotate', 'left-to-right', 'near-to-far']
    # Process only the first folder for now
    for folder in folders1[0:1]:
        # Path to the data folder
        data_path = os.path.join('./results', folder)
        # Process the video in the folder
        process_single_video(data_path) 