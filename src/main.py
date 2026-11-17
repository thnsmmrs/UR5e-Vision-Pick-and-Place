import rtde_control
import rtde_receive
import rtde_io
import numpy as np
import IK
import waypoints
import vision
import centroid
import transform

ROBOT_IP = "172.16.34.2"
GRIPPER_PORT = 63352

rtde_c = rtde_control.RTDEControl(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceive(ROBOT_IP)
rtde_io = rtde_io.RTDEIO(ROBOT_IP)



u,v,h,color,depth = vision.get_info()

target_pose = transform(u,v)
joint_vars = IK(rtde_c, rtde_r, target_pose)

rtde_c.moveJ(joint_vars, 0.5, 0.1)