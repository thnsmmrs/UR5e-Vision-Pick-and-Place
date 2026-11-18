import pyrealsense2 as rs
import numpy as np
import cv2
import centroid

#RealSense documentation
#https://github.com/realsenseai/librealsense/tree/master/wrappers/python#table-of-contents
#https://dev.realsenseai.com/


def start_cam():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    return pipeline, align, depth_scale

def calibrate_bg(pipeline, align, depth_scale):
    background_frames = []

    for i in range(30):
        frames = align.process(pipeline.wait_for_frames())
        depth = frames.get_depth_frame()

        if depth:
            depth_image = np.asanyarray(depth.get_data())
            depth_m = depth_image.astype(np.float32) * depth_scale
            depth_m[depth_image == 0] = np.nan

            background_frames.append(depth_m)

    background = np.nanmedian(background_frames, axis=0)

    return background

def detect_frame(pipeline, align, depth_scale, background, color):
    #capture RGB and depth frames
    frames = align.process(pipeline.wait_for_frames())
    rgb = frames.get_color_frame()
    depth = frames.get_depth_frame()


    if not rgb or not depth:
        return None

    color_image = np.asanyarray(rgb.get_data())
    depth_image = np.asanyarray(depth.get_data())

    #get bounds and centroid from RGB color detection
    result = centroid.detect_object(color_image, color)
    if result is None:
        return None

    (u, v), bounds = result
    x, y, w, h = bounds

    #calculate height above background table
    depth_m = depth_image.astype(np.float32) * depth_scale
    height_map = background - depth_m

    #sample depth near center of object to avoid edges
    x1 = x + w // 4
    x2 = x + 3 * w // 4
    y1 = y + h // 4
    y2 = y + 3 * h // 4

    heights = height_map[y1:y2, x1:x2]

    #ignore invalid readings and heights outside expected range
    heights = heights[
        np.isfinite(heights) &
        (heights > 0.005) &
        (heights < 0.070)
    ]

    if len(heights) == 0:
        return None

    #median reduces noise from individual depth measurements
    object_height = np.median(heights)

    #draw centroid and bounding box on RGB frame
    display = color_image.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.circle(display, (u, v), 5, (0, 255, 255), -1)

    #create depth visualization
    height_8bit = (
        np.clip(np.nan_to_num(height_map, nan=0) / 0.060, 0, 1) * 255
    ).astype(np.uint8)

    depth_display = cv2.applyColorMap(height_8bit, cv2.COLORMAP_JET)

    #save frames for optional display in main
    debug_frames = {
        "color": display,
        "depth": depth_display
    }

    return (u, v), object_height, debug_frames

def get_info(pipeline, align, depth_scale, background, color,
             samples=20, tolerance=10):

    measurements = []

    #collect 20 valid measurements, allowing some failed detections
    for i in range(samples * 3):
        result = detect_frame(pipeline, align, depth_scale, background, color)

        if result is not None:
            measurements.append(result)

        if len(measurements) == samples:
            break

    if not measurements:
        return None

    #find median centroid to compare measurements against
    positions = np.array([m[0] for m in measurements])
    median = np.median(positions, axis=0)

    #remove detections more than 10 pixels from median
    distances = np.linalg.norm(positions - median, axis=1)

    good = [
        m for m, d in zip(measurements, distances)
        if d <= tolerance
    ]

    #make sure enough measurements are consistent
    if len(good) < max(5, samples // 2):
        return None

    #average remaining centroids and heights
    avg_position = np.mean([m[0] for m in good], axis=0)
    avg_height = np.mean([m[1] for m in good])

    u, v = np.rint(avg_position).astype(int)

    #keep frames from last accepted measurement
    frames = good[-1][2]

    return (int(u), int(v)), float(avg_height), frames


