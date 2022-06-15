# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# cv2.so: undefined symbol: PyCObject_Type
import cv2
import numpy as np
import time
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
cv2.namedWindow('capture')
fps = 25
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter=cv2.VideoWriter('./2021.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
success, frame = cap.read()

while success and cv2.waitKey(1) == -1:
    videoWriter.write(frame)
    image = cv2.resize(frame,(864,576))
    cv2.imshow('capture', image)
    time.sleep(0.04)
    success, frame = cap.read()

cv2.destroyWindow('capture')
cap.release()
