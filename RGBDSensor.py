import pyrealsense2 as rs
import numpy as np
import open3d as o3d

class RGBDSensor:
    def __init__(self):
        # Create a RealSense pipeline object
        self.pipeline = rs.pipeline()

        # Create a RealSense configuration object and set the resolution and frame rate
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        
        # Start the pipeline and get the color and depth sensor objects
        profile = self.pipeline.start(config)

        # Get the intrinsic parameters of the depth stream
        depth_stream = profile.get_stream(rs.stream.depth)
        color_stream = profile.get_stream(rs.stream.color)
        
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_to_color_extrin = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
        print(self.depth_intrinsics, self.color_intrinsics, self.depth_to_color_extrin)


        # # Print the intrinsic parameters
        # print("Intrinsic Parameters:")
        # print("Width:", intrinsics.width)
        # print("Height:", intrinsics.height)
        # print("PPX (Principal Point X):", intrinsics.ppx)
        # print("PPY (Principal Point Y):", intrinsics.ppy)
        # print("FX (Focal Length X):", intrinsics.fx)
        # print("FY (Focal Length Y):", intrinsics.fy)
        # print("Distortion Model:", intrinsics.model)
        # print("Distortion Coefficients:", intrinsics.coeffs)

        
        # Set the exposure and gain values for the color sensor
        self.color_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        self.color_sensor.set_option(rs.option.exposure, 1000)
        self.color_sensor.set_option(rs.option.gain, 32)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)





    def get_images(self):
        # Wait for RealSense data streams
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        # If there are no valid frames, continue waiting for the next one
        if not color_frame or not depth_frame:
            return None, None
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        # Convert color and depth data to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image, depth_frame

    def project_point_to_pixel(self, intrinsics, point):
        print(point)
        print(intrinsics.ppy,intrinsics.ppx)
        print(intrinsics.fy,intrinsics.fx)
        x = point[0] / point[2] # x / z
        y = point[1] / point[2] # y / z
        u = round(x * intrinsics.fx + intrinsics.ppx)
        v = round(y * intrinsics.fy + intrinsics.ppy)
        return (v,u)

    def get_point_cloud(self, depth_frame):
        pc = rs.pointcloud()
        points = rs.points()
        points = pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2))
        w = depth_frame.get_width() # 640 
        depth_pcd = np.reshape(vertices,(-1,w,3)) # (480,640,3)
        # depth_pcd[:, :, 1] = -depth_pcd[:, :, 1] 
        #pixel = self.project_point_to_pixel(self.color_intrinsics, depth_pcd[479,639]) # 0,0,0 = 【240,320】
        #print("Pixel coordinates:", pixel)
        #pixel1 = self.project_point_to_pixel(self.color_intrinsics, depth_pcd[400,600]) # 0,0,0 = 【240,320】
        #print("Pixel1 coordinates:", pixel1)
        # import pdb; pdb.set_trace()
        #vertices_all = depth_pcd[:,:,:].reshape(-1,3)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(vertices_all) 
        #o3d.visualization.draw_geometries([pcd])
        
        return depth_pcd

    def release(self):
        self.pipeline.stop()