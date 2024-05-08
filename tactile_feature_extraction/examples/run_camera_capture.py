import cv2
import time 
from collections import deque
import numpy as np

source = 0
cam = cv2.VideoCapture(source)

if cam is None or not cam.isOpened():
    print('Warning: unable to open video source: ', source)

# Auto exposure
flag = cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
print('Set auto exposure mode, ', flag)

# Resolution
flag = cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
flag = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('Set resolution, ',flag)

# Exposure
flag = cam.set(cv2.CAP_PROP_EXPOSURE, -5)
print('Set exposure, ', flag)

flag = cam.set(cv2.CAP_PROP_BRIGHTNESS, 64)
print('Set brightness, ', flag)

def process_image(image):
    # Convert to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add channel axis
    image = image[..., np.newaxis]

    # image = image.astype('uint8')
    # image = cv2.medianBlur(image, 19)

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, -5)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 79, -3)
    return image
    

prev_frame_time = 0
new_frame_time = 0
running_fps = deque(maxlen=30)

while True:
    success, frame = cam.read()
    frame = process_image(frame)

    # Time of reading the frame
    new_frame_time = time.time()

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    running_fps.append(fps)
    prev_frame_time = new_frame_time
    fps_str = "FPS: {:.2f}".format(sum(running_fps)/len(running_fps))

    cv2.putText(frame, fps_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("image", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()