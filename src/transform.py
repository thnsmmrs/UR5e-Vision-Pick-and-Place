#transforming pixel space (u,v) to world space (x,y) 
#z to be output from depth camera
#we could explore homography transformation, exploring fixed height linear transform

def pixel_to_world(u, v):
    cam_w = 1280 #px
    cam_h = 720 #px from 1080p resolution
    cam_height = 1.68 #meters

    

    x=1
    y=1
    z=1
    theta=1
    return (x,y,z, theta)