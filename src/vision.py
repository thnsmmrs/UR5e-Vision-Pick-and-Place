import pyrealsense2 as rs
import numpy as np
import cv2
import XY

#RealSense documentation
#https://github.com/realsenseai/librealsense/tree/master/wrappers/python#table-of-contents
#https://dev.realsenseai.com/

#initialize cam pipeline and configure streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

pipeline.start(config)
# Capture background depth reference
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

background_frames = []

for i in range(30):
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()

    if depth:
        depth_image = np.asanyarray(depth.get_data()).copy()
        depth_m = depth_image.astype(np.float32) * depth_scale
        depth_m[depth_image == 0] = np.nan
        background_frames.append(depth_m)

# Median across 30 frames reduces random noise
background = np.nanmedian(
    np.stack(background_frames), axis=0
)

depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
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
        # Calculate current depth in meters
        depth_m = depth_image.astype(np.float32) * depth_scale

        # Subtract background
        height_map = background - depth_m

        # Generate depth colormap based on height above table
        height_normalized = np.clip(height_map / 0.060, 0, 1)
        height_8bit = (height_normalized * 255).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(
            height_8bit, cv2.COLORMAP_JET
        )

        # Set invalid pixels to black
        valid = (depth_image > 0) & np.isfinite(background)
        depth_colormap[~valid] = 0
        # Identify pixels 12–60 mm above the background table
        mask = (
            valid
            & (height_map > 0.012)
            & (height_map < 0.060)
        ).astype(np.uint8) * 255
        # Define workspace ROI (region of interest)
        h, w = mask.shape

        x_min = int(w * 0.29)
        x_max = int(w * 0.69)
        y_min = 0
        y_max = int(h * 0.55)

        # Create workspace mask
        workspace_mask = np.zeros_like(mask)
        workspace_mask[y_min:y_max, x_min:x_max] = 255

        # Keep only detections inside workspace
        mask = cv2.bitwise_and(mask, workspace_mask)
        # Remove small noise regions
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Select largest detected object
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) > 20:
                # Calculate centroid using image moments
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 4, (0, 0, 255), -1)
        # Create filled object mask
        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)

        # Create a ring outside the object
        kernel = np.ones((11, 11), np.uint8)
        outer_mask = cv2.dilate(object_mask, kernel)
        table_ring = cv2.subtract(outer_mask, object_mask)

        # Sample valid depth measurements (meters)
        valid = depth_image > 0

        cube_depths = depth_m[(object_mask > 0) & valid]
        table_depths = depth_m[(table_ring > 0) & valid]

        if len(cube_depths) > 0 and len(table_depths) > 0:
            cube_depth = np.median(cube_depths)
            table_depth = np.median(table_depths)

            # Approximate cube height
            cube_height = table_depth - cube_depth

            # Approximate depth of cube's geometric center
            center_depth = table_depth - cube_height / 2

            print(f"Centroid: ({cx}, {cy})")
            print(f"Cube height: {cube_height * 1000:.1f} mm")
            print(f"Center depth: {center_depth:.4f} m")
        cv2.imshow("Height Above Table", depth_colormap)

        #show both windows through opencv
        #cv2.imshow("RealSense Depth", depth_colormap)
        cv2.rectangle(
            color_image,
            (x_min, y_min),
            (x_max - 1, y_max - 1),
            (255, 0, 0),
            2
        )
        cv2.imshow("RealSense Color", color_image)
        point = XY.detect_object(color_image, "Red")

        
        #close windows when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
