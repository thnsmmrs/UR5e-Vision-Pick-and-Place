import rtde_control
import rtde_receive
import rtde_io
import numpy as np
import cv2
import IK, waypoints, vision, centroid, transform



ROBOT_IP = "172.16.34.2"
GRIPPER_PORT = 63352

#rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
#rtde_r = rtde_receive.RTDEReceive(ROBOT_IP)
#rtde_io = rtde_io.RTDEIO(ROBOT_IP)

pipeline, align, depth_scale = vision.start_cam()
background = vision.calibrate_bg(pipeline, align, depth_scale)
#color = input("Choose Red / Green / Blue: ").strip().capitalize()
color = "Green"
input("")
result = vision.get_info(pipeline, align, depth_scale, background, color)

if result is None:
    print("Object detection failed")
else:
    (u, v), h, frames = result

    print(f"Centroid: ({u}, {v})")
    print(f"Height: {h * 1000:.1f} mm")

    cv2.imshow("Color", frames["color"])
    cv2.waitKey(0)
pipeline.stop()
cv2.destroyAllWindows()
#target_pose = transform(u,v)
#joint_vars = IK(rtde_c, rtde_r, target_pose)

#rtde_c.moveJ(joint_vars, 0.5, 0.1)