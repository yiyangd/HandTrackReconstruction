import os
import pdb
import time
import cv2
import numpy as np
import socket
# Import functions for fitting and manipulating the hand model
from fit_hand_model_modify import get_axis_from, fit_hand_model, foward_hand_model, get_init_parameters, get_parameter_bounds, minimize, print_pred_results
import matplotlib.pyplot as plt
from glob import glob
from hand_landmark_estimation import estimate_hand_lengths, get_initial_hand_lengths, estimate_hand_orientation, compute_local_frames_by_pts

# Function to plot the global coordinate frame
def plot_global_frame(ax, T, length=0.02):
    """
    Plots the global coordinate frame using quiver arrows.

    Args:
        ax: Matplotlib 3D axis.
        T: Transformation matrix (rotation and translation).
        length: Length of the arrows representing the coordinate axes.
    """
    R = T[:3, :3].T # Extract rotation matrix
    p = T[:3, 3]    # Extract translation vector
    # Plot quiver for each axis (x: red, y: green, z: blue)
    for i, color in enumerate(['r', 'g', 'b']):
        ax.quiver(p[0], p[1], p[2], R[i, 0], R[i, 1], R[i, 2], color=color, length=length)

# Function to visualize the 3D hand landmarks
def visualize_landmarks_3d(m_landmarks,image, global_frames, outliers):
    """
    Visualizes the hand landmarks in 3D.

    Args:
        m_landmarks: Modeled landmarks.
        p_landmarks: Predicted landmarks.
        image: Input image for visualization.
        global_frames: Dictionary of global frames for different parts of the hand.
        outliers: Array of outliers detected in the hand landmarks.
    """
    fig = plt.figure()
    ax0 = fig.add_subplot(121) # 2D plot of the image
    ax0.imshow(image)
    ax = fig.add_subplot(122, projection='3d') # 3D plot for landmarks

    # Define connections between landmarks to visualize the hand structure
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    # # Extract x, y, z coordinates
    # xs = p_landmarks[:, 0]
    # ys = p_landmarks[:, 1]
    # zs = p_landmarks[:, 2]

    # # Plot predicted landmarks and their connections
    # ax.scatter(xs, ys, zs, c='m', marker='*')
    # for connection in connections:
    #     start_idx, end_idx = connection
    #     ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], [zs[start_idx], zs[end_idx]], 'm')
    
    # Extract x, y, z coordinates of modeled landmarks
    # m_landmarks = m_landmarks[outliers < 0.5]
    xs = m_landmarks[:, 0]
    ys = m_landmarks[:, 1]
    zs = m_landmarks[:, 2]

    # Plot modeled landmarks and their connections
    ax.scatter(xs, ys, zs, c='k', marker='o')
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], [zs[start_idx], zs[end_idx]], 'k')
    
    # Plot global frames for different parts of the hand (e.g., wrist, fingers)
    if True:
        for i in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            plot_global_frame(ax, global_frames[f'T_c_{i}'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

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
    eig_values.sort() # Sort eigenvalues
    if eig_values[1] < 1e-2:
        return 0. # Low uncertainty if eigenvalues are small
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
    distances = np.sqrt(np.sum((p-median_xyz[None])**2, -1))
    # Detect large deviations as outliers
    outliers0 = (distances > 0.2).astype(np.float32)

    # Find non-obvious false measurements
    # Further refine outliers based on uncertainties for different hand parts
    confidence1 = compute_uncertainty_matrix(p[1:5]) # 1-4
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

    outliers0 = np.clip(outliers0, a_min=0, a_max=1.) # Clip values to [0, 1]

    return outliers0

# Function to process a single frame of hand tracking
def process_single_frame(p, uv, x, lengths):
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
    outliers = outlier_detection(p) # Detect outliers

    # If initial parameters not provided, generate them randomly
    if x is None:
        x = get_init_parameters(p, random=True)

    # Set base position of hand model
    x[28:] = p[0]
    # Optimize the hand model parameters using Powell method
    result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths, outliers), bounds=get_parameter_bounds(), method="Powell", options={'maxiter': 100, })
    #import pdb; pdb.set_trace()
    # gtol 梯度范数上限，norm 范数浮动上限， eps 求解浮动上限
    # result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths),
    #                   bounds=get_parameter_bounds(), method="bfgs",
    #                   options={'gtol': 1e-3, 'eps': 1e-3})

    # result = minimize(fit_hand_model, x0=x, args=(p, uv, lengths, outliers),
    #                   bounds=get_parameter_bounds(), method="nelder-mead",
    #                   options={'gtol': 1e-9, 'eps': 1e-9})

    print(f'Global Minimum: {result.fun}') # Print the final error of the optimization
    # print(f'x = {result.x}')
    # Use the optimized parameters to predict the hand pose
    pred_x = result.x # Get the predicted model parameters
    # Predict 3D points and frames
    pred_uv, pred_p, global_frames, data_list = foward_hand_model(pred_x, lengths)


    # fig = plt.figure()
    # ax0 = fig.add_subplot(1, 2, 1)
    # ax0.imshow(img)
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.plot(p[:, 0], p[:, 1], p[:, 2], "*")
    # ax.plot(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], ">")


    # axis_0 = get_axis_from(T_c_0, pred_p[0], scale=0.01)
    #
    # ax.plot([axis_0[0, 0, 0], axis_0[0, 0, 1]], [axis_0[0, 1, 0], axis_0[0, 1, 1]], [axis_0[0, 2, 0], axis_0[0, 2, 1]],
    #         'r-')
    # ax.plot([axis_0[1, 0, 0], axis_0[1, 0, 1]], [axis_0[1, 1, 0], axis_0[1, 1, 1]], [axis_0[1, 2, 0], axis_0[1, 2, 1]],
    #         'g-')
    # ax.plot([axis_0[2, 0, 0], axis_0[2, 0, 1]], [axis_0[2, 1, 0], axis_0[2, 1, 1]], [axis_0[2, 2, 0], axis_0[2, 2, 1]],
    #         'b-')
    # plt.show()
    return pred_x, pred_p[:, :3], global_frames, outliers, data_list

# Function to process a video sequence of hand motions and fit the hand model to each frame
def process_single_video(data_path, load_pre_save_data=False, visualize=False, save_results=False):
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
    b = [25,60,130,160,170,195]
    b = [25,40,50,60,70, 130,140, 150, 160,170, 180, 195]
    # Initialize variables for average processing time calculation
    total_processing_time = 0
    total_frames = 0
    # Process frames with a step of 5
    for frame_i in range(0, n_frame, 5):
        # Path to the image frame
        # if  (frame_i not in a):
        #     continue
        img_path = os.path.join(data_path, f'annotate_img_{frame_i}.png')
        # Path to the landmarks data
        kpts_path = os.path.join(data_path, f'data_{frame_i}.txt')
        # Read the image
        img = cv2.imread(img_path)
        # Load the landmark points
        kpts = np.loadtxt(kpts_path).astype(np.float32)
        # Skip frame if landmarks are incomplete
        if kpts.shape[0] != 21:
            continue
        p = kpts[:, 2:] # 3D points (x, y, z)
        p[:, 1] = -p[:, 1] 
        uv = kpts[:, :2]  # 2D image coordinates
        # Display the image
        cv2.imshow("Hand", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Stop the video if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        outliers = outlier_detection(p)  # Detect outliers
        # import pdb; pdb.set_trace()
        # Start time for measuring processing time
        start_time = time.time()

        pred_x, pred_p, global_frames, outliers, _, err, data_list = compute_local_frames_by_pts(p, uv, outliers)
        # End time after processing
        end_time = time.time()
        total_frames += 1
        # Calculate time taken per frame and accumulate
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_processing_time += processing_time
        print(f"Processing time for frame {frame_i}: {processing_time:.2f} ms")
        # Append predicted root point to data list0
        data_list.extend([p[0, 0], -p[0, 1], p[0, 2]])
        
        # if visualize:
        #     visualize_landmarks_3d(p,p, img, local_frames, outliers)
        # Send data via UDP
        sock.sendto(str.encode(str(data_list)), serverAddressPort)
        
        # Delay to simulate real-time processing
        time.sleep(1)
        
    # Calculate and display average processing time
    if total_frames > 0:
        average_processing_time = total_processing_time / total_frames
        print(f"Total processing time per frame: {total_processing_time:.2f} ms")
        print(f"Total Frames: {total_frames}")
        print(f"Average processing time per frame: {average_processing_time:.2f} ms")
    # Release resources and stop the pipeline
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # List of experiment folders
    folders1 = ['grasp','rotate_z','finger-rotate', 'left-to-right', 'near-to-far']
    folders2 = ['prediction_70']
    # Process only the first folder for now
    for folder in folders1[0:1]:
        # Path to the data folder
        data_path = os.path.join('./results', folder)
        # Process the video in the folder
        process_single_video(data_path) 
