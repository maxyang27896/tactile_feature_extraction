from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController
import time
from ipdb import set_trace
import math

tcp = 65
offset = 0              # For tips 1-4
# offset = 6              # For tips 5 and 6
base_frame = (0, 0, 0, 0, 0, 0)  
# base frame: x->front, y->right, z->up (higher z to make sure doesnt press into the table)
work_frame = (473, -111, 60.75-offset, -180, 0, -90)            # For 0 degrees
# work_frame = (473, -111, 66.75-offset, -180, 0, -90)            # For tips 5 and 6 (45 degrees)
# work_frame = (473, -40, 61-offset, -180, 0, -90)           # safe baseframe for testing, using a box
tcp_x_offset = -1.5                 # For tips 1-4
# tcp_x_offset = -1.75                 # For tips 5 and 6
tcp_y_offset = 1.5

with AsyncRobot(SyncRobot(RTDEController(ip='192.11.72.20'))) as robot:
    time.sleep(1)
    tcp_x = tcp_x_offset*math.sin(math.pi/4) + tcp_y_offset*math.cos(math.pi/4)
    tcp_y = tcp_x_offset*math.cos(math.pi/4) - tcp_y_offset*math.sin(math.pi/4)

    robot.tcp = (tcp_x, tcp_y, tcp + offset-0.25, 0, 0, -45) # 60 if tcp at the center of the hemisphere, otherwise 75 is keeping the tcp on the skin
    # 85mm is true tcp but when rotating at large angles, it might crush the sensor
    robot.axes = "sxyz"
    robot.linear_speed = 100
    robot.angular_speed = 100
    robot.coord_frame = work_frame
    set_trace()
    robot.move_linear((0, 0, 0, 0, 0, 0)) #move to home position
    print('Moved to home position')
    set_trace()

    # Test ranges
    try:
        while True:
            robot.linear_speed = 30
            robot.move_linear((0, 0, 0, -28, 0, 0)) # Moved to x rotation lower
            robot.move_linear((0, 0, 0, 0, 0, 0))
            robot.move_linear((0, 0, 0, +28, 0, 0)) # Moved to x rotation higher
            robot.move_linear((0, 0, 0, 0, 0, 0))
            set_trace()
    except:
        print("Continung to y rotation")

    try:
        while True: 
            robot.linear_speed = 30
            robot.move_linear((0, 0, 0, 0, -28, 0)) # Moved to y rotation lower
            robot.move_linear((0, 0, 0, 0, 0, 0))
            robot.move_linear((0, 0, 0, 0, 28, 0)) # Moved to y rotation higher
            robot.move_linear((0, 0, 0, 0, 0, 0))
            set_trace()
    except:
        print('Moving to safe location')
    
    robot.linear_speed = 30
    robot.move_linear((0, 0, -50, 0, 0, 0)) #move to a bit higher position to avoid damaging the sensor