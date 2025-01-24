import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green




class HandLandmarker:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.1
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, image):
        # Create MediaPipe image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        ) # mp_image.height = 480, mp_image.width = 640
        # Perform hand landmarks detection on the provided single image.
        # The hand landmarker must be created with the image mode.
        return self.landmarker.detect(mp_image)



def draw_landmarks_on_image(rgb_image, hand_landmarker_result):
    """
    Draws landmarks and handedness on the image.

    Args:
        rgb_image: The original RGB image.
        hand_landmarker_result: The result from hand landmark detection.

    Returns:
        Annotated image with landmarks and handedness text.
    """
    # Extracts the landmarks of the detected hands 
    # and their handedness from the detection result.
    hand_landmarks_list = hand_landmarker_result.hand_landmarks
    handedness_list = hand_landmarker_result.handedness
    #print(hand_landmarks_list)
    # Creates a copy of the original image on which we will draw the landmarks.
    annotated_image = np.copy(rgb_image)
    # Checks if any hand landmarks were detected
    if len(hand_landmarks_list) == 0:
        #annotated_image = cv2.flip(annotated_image,1)
        return annotated_image
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        # Extracts the landmarks and handedness for the current hand
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
    # The image is flipped along the vertical axis before being returned 
    # to make it more visually intuitive for the viewer (as if they're looking in a mirror).
    # annotated_image = cv2.flip(annotated_image, 1)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        height, width, _ = annotated_image.shape
        # Calculates the top left corner of the bounding box of the hand in the image. 
        # The top left corner is calculated using the maximum x-coordinate and minimum y-coordinate among the hand landmarks.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        #if idx == 0:
            #print(x_coordinates)
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = width - int(max(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Puts the handedness text at the calculated position in the image 
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)     
    

    return annotated_image

def visualize_landmarks_3d(landmarks):
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
    xs = landmarks[0::3]
    ys = landmarks[1::3]
    zs = landmarks[2::3]
    
    # Plot each landmark
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # Plot connections
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], [zs[start_idx], zs[end_idx]], 'b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()