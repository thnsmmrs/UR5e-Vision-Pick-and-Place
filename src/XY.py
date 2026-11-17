#theoretical working implementation with frame capture
import cv2


def detect_object(frame, color, region):
    x, y, w, h = region

    roi = frame[y:y + h, x:x + w]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if color == "Red":
        mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 50), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "Green":
        mask = cv2.inRange(hsv, (35, 70, 50), (85, 255, 255))

    elif color == "Blue":
        mask = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))

    else:
        return None

    M = cv2.moments(mask, binaryImage=True)

    if M["m00"] < 300:
        return None

    u = int(M["m10"] / M["m00"]) + x
    v = int(M["m01"] / M["m00"]) + y

    return (u, v)


cap = cv2.VideoCapture(0)

ok, frame = cap.read()
cap.release()

if not ok:
    print("Camera failed")
    exit()

region = cv2.selectROI("Select Workspace", frame)
cv2.destroyAllWindows()

if region[2] == 0 or region[3] == 0:
    print("No region selected")
    exit()

objects = {}

for color in ["Red", "Green", "Blue"]:
    point = detect_object(frame, color, region)

    if point is not None:
        objects[color] = point

if not objects:
    print("No objects detected")
    exit()

colors = list(objects.keys())

print("\nAvailable Objects:")

for i in range(len(colors)):
    print(f"{i + 1}. {colors[i]}")

choice = int(input("\nSelect an object: "))

if choice < 1 or choice > len(colors):
    print("Invalid selection")
    exit()

color = colors[choice - 1]
point = objects[color]

print(f"\nSelected: {color}")
print(f"Centroid: {point}")
