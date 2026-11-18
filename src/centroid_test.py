
import cv2
import numpy as np
import pyrealsense2 as rs


def detect_object(frame, color):
    h, w = frame.shape[:2]

    # Custom workspace cropping
    x1, x2 = int(w * 0.29), int(w * 0.69)
    y1, y2 = 0, int(h * 0.55)

    roi = frame[y1:y2, x1:x2]

    # Color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if color == "Red":
        mask1 = cv2.inRange(hsv, (0, 110, 60), (8, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 110, 60), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "Green":
        mask = cv2.inRange(hsv, (35, 70, 50), (85, 255, 255))

    elif color == "Blue":
        mask = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))
        cv2.imshow("Blue Mask", mask)

    else:
        return None

    # Find largest contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 30:
        return None

    # Calculate centroid
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    u = int(M["m10"] / M["m00"]) + x1
    v = int(M["m01"] / M["m00"]) + y1

    return (u, v)


# MAIN PROGRAM
#color = input("Choose Red / Green / Blue: ").strip().capitalize()
color = "Blue"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        rgb = frames.get_color_frame()

        if not rgb:
            continue

        frame = np.asanyarray(rgb.get_data()).copy()

        # Detect centroid
        point = detect_object(frame, color)

        # Draw workspace
        h, w = frame.shape[:2]
        cv2.rectangle(
            frame,
            (int(w * 0.29), 0),
            (int(w * 0.69), int(h * 0.55)),
            (255, 0, 0), 2
        )

        if point is not None:
            u, v = point
            cv2.circle(frame, (u, v), 6, (0, 255, 255), -1)
            cv2.putText(
                frame, f"({u}, {v})",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1
            )

        cv2.imshow("RGB Centroid Test", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
