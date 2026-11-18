
import cv2


def detect_object(frame, color):
    h, w = frame.shape[:2]

    #custom workspace cropping to avoid noise generation
    x1, x2 = int(w * 0.29), int(w * 0.69)
    y1, y2 = 0, int(h * 0.55)

    roi = frame[y1:y2, x1:x2]

    #color segmentation, based on predefined hue range
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if color == "Red":
        mask1 = cv2.inRange(hsv, (0, 110, 60), (8, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 110, 60), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "Green":
        mask = cv2.inRange(hsv, (35, 70, 50), (85, 255, 255))

    elif color == "Blue":
        mask = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))

    else:
        return None

    #finding largest contour from colors
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 30: #30px minimum to avoid noise - change if needed
        return None

    #find centroid from largest contour
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    u = int(M["m10"] / M["m00"]) + x1
    v = int(M["m01"] / M["m00"]) + y1
    # Get bounding box of detected contour, used for search
    x, y, w, h = cv2.boundingRect(contour)

    # Convert coordinates from cropped ROI to full image
    bounds = (x + x1, y + y1, w, h)

    return (u, v), bounds
