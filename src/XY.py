
import cv2
import numpy as np


def detect_object(frame, color):
    h, w = frame.shape[:2]

    # Define workspace bounds
    # X: 30-70% of image width
    # Y: 0-65% of image height
    x_min = int(w * 0.30)
    x_max = int(w * 0.70)
    y_min = 0
    y_max = int(h * 0.65)

    # Convert full image to HSV
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Identify selected color
    if color == "Red":
        mask1 = cv2.inRange(hsv, (0, 60, 40), (15, 255, 255))
        mask2 = cv2.inRange(hsv, (165, 60, 40), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "Green":
        mask = cv2.inRange(hsv, (35, 70, 50), (85, 255, 255))

    elif color == "Blue":
        mask = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))

    else:
        return None

    # Create full-frame workspace mask
    workspace_mask = np.zeros((h, w), dtype=np.uint8)
    workspace_mask[y_min:y_max, x_min:x_max] = 255

    # Ignore colors outside workspace
    mask = cv2.bitwise_and(mask, workspace_mask)

    # Calculate image moments
    M = cv2.moments(mask, binaryImage=True)

    if M["m00"] < 300:
        return None

    # Full-frame centroid coordinates
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])

    return (u, v)
