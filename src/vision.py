import pyrealsense2 as rs
import numpy as np
import cv2

#RealSense documentation
#https://github.com/realsenseai/librealsense/tree/master/wrappers/python#table-of-contents
#https://dev.realsenseai.com/

#initialize cam pipeline and configure streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

#start streaming and receiving frames
try:
    while True:
        frames = pipeline.wait_for_frames()

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if not color or not depth:
            continue

        #convert to numpy arrays
        color_image = np.asanyarray(color.get_data())
        depth_image = np.asanyarray(depth.get_data())

        #convert depth measurements to visible colors
        depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        #show both windows through opencv
        cv2.imshow("RealSense Depth", depth_colormap)
        cv2.imshow("RealSense Color", color_image)

        #opencv frame analysis here
        
        #close windows when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
