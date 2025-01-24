import cv2
import numpy as np
import socket
import os
from RGBDSensor import *
from mp_utils import HandLandmarker, draw_landmarks_on_image, visualize_landmarks_3d
import time
# from o3d_visualization import visualize_landmarks
#from YOLOv8 import *
#import datetime as dt

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)



# ======================================================================
sensor = RGBDSensor() # RealSense Configuration
landmarker = HandLandmarker()

files = os.listdir('./results/')
path = f'./results/prediction_{len(files)+1}'
os.makedirs(path, exist_ok=True)

save_info = True
frame_id = 0
# =======================================================================
# Initialize variables for average processing time calculation
total_processing_time = 0

# Create a loop to read the latest frame from the sensor
while True:
    color_image, depth_image, depth_frame = sensor.get_images()
    (h,w,_) = color_image.shape #  height 480 and width 640 of the image
    depth_pcd = sensor.get_point_cloud(depth_frame)

    # The following code is to convert y from Mediapipe Frame to Unity Frame
    depth_pcd[:, :, 1] = -depth_pcd[:, :, 1] 

    # Convert color_image from frame to a MediaPipe’s Image object.
    # color_image = cv2.flip(color_image,1)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Start time for measuring processing time
    start_time = time.time()

    # Detect hand landmarks
    hand_landmarker_result = landmarker.detect(color_image_rgb)

    # End time after processing
    end_time = time.time()

    # Calculate time taken per frame and accumulate
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    total_processing_time += processing_time
    print(f"Processing time for frame {frame_id}: {processing_time:.2f} ms")
    #import pdb; pdb.set_trace()
    #if (hand_landmarker_result.hand_world_landmarks):
    #    print(hand_landmarker_result.hand_world_landmarks[0][0].x)
    data = []
    uvxyz = []

    if len(hand_landmarker_result.hand_landmarks) > 0:
        landmarks = hand_landmarker_result.hand_landmarks[0]
        #import pdb; pdb.set_trace()
        for i_mark, landmark in enumerate(landmarks): # iterate 21 landmarks
            
            img_x, img_y = landmark.x, landmark.y
            img_x, img_y = int(img_x * w) , int(img_y * h)
            
            if 0 <= img_x < w and 0 <= img_y < h:
                #print(img_x,", ", img_y, ", ", depth_pcd[img_y, img_x][0], ", ", depth_pcd[img_y, img_x][1], ", ", depth_pcd[img_y, img_x][2])
                
                uvxyz.extend([[img_x,img_y,depth_pcd[img_y, img_x][0],depth_pcd[img_y, img_x][1],depth_pcd[img_y, img_x][2]]])
                data.extend(depth_pcd[img_y, img_x].tolist())
            else:
                data.extend([0,0,0])
        if uvxyz:
            print(uvxyz[9])
        data = [round(x * 100,2) for x in data]
        sock.sendto(str.encode(str(data)), serverAddressPort)
    else:
        data = [0] * 63
        uvxyz = [0] * 105
        sock.sendto(str.encode(str(data)), serverAddressPort)
    

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(color_image_rgb, hand_landmarker_result)

    if save_info:
        cv2.imwrite(path+f'/annotate_img_{frame_id}.png', annotated_image)
        np.savetxt(path+f'/data_{frame_id}.txt', np.array(uvxyz))
        frame_id += 1


    # Visualize the hand landmarks in 3D
    # if len(hand_landmarker_result.hand_landmarks) > 0:
    #     import pdb; pdb.set_trace()
    #     visualize_landmarks_3d(data)
    #annotated_image = cv2.flip(annotated_image,1)

    cv2.imshow("Hand", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    #cv2.imshow("Image", color_image_rgb)
    if cv2.waitKey(1) == ord('q'):
        break

# Calculate and display average processing time
if frame_id > 0:
    average_processing_time = total_processing_time / frame_id
    print(f"Total processing time per frame: {total_processing_time:.2f} ms")
    print(f"Total Frames: {frame_id}")
    print(f"Average processing time per frame: {average_processing_time:.2f} ms")

# Release resources and stop the pipeline
cv2.destroyAllWindows()
sensor.release()
    
