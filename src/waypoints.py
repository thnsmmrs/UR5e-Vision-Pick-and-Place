import rtde_control

def go_home(robot):
    robot.moveJ([-2.1e-05, -1.570804, 5e-06, -1.570813, 1e-06, 1.5e-05], 0.9, 0.3)
    