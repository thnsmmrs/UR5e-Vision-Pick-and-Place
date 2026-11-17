def calculate_ik(rtde_c, rtde_r, target_pose):
    current_joints = rtde_r.getActualQ()

    try:
        joint_angles = rtde_c.getInverseKinematics(
            target_pose,
            current_joints
        )

        return joint_angles

    #error handling
    #troubleshooting:
    # 1. check if target is reachable by UR5e
    # 2. verify no errors on UR5e pendant
    # 3. verify that the UR5e is in remote control
    except RuntimeError as e:
        print(f"IK calculation failed: {e}")
        return None