import cv2
import numpy as np
import os

cap = cv2.VideoCapture("D:\\Users\\nole\\Desktop\\video\\6.mp4")
# cap = cv2.VideoCapture(1)
# cv2.namedWindow('capture')
# fps = 30
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter=cv2.VideoWriter('./video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
success, frame = cap.read()
cnt = 0
path = 'D:\\Users\\nole\\Desktop\\video\\6'
if not os.path.exists(path):
    os.makedirs(path)

while success and cv2.waitKey(1) == -1:
    # videoWriter.write(frame)
    # cv2.imshow('capture', frame)
    if cnt % 5 == 0:
        imgs = path + '/9{:0>5d}.jpg'.format(cnt)
        cv2.imwrite(imgs, frame)
    cnt += 1
    success, frame = cap.read()

# video0102
# while success and cv2.waitKey(1) == -1:
#     # videoWriter.write(frame)
#     # cv2.imshow('capture', frame)
#     if cnt >= 20:
#         if cnt < 370:
#             if (cnt-20) % 3 == 0:
#                 if ((cnt-20) / 3) % 4 == 3:
#                     imgs = path + '/01{:0>4d}.jpg'.format(cnt-20)
#                     cv2.imwrite(imgs, frame)
#         if cnt >= 430:
#             if (cnt-80) % 3 == 0:
#                 if ((cnt-80) / 3) % 4 == 3:
#                     imgs = path + '/01{:0>4d}.jpg'.format(cnt-80)
#                     cv2.imwrite(imgs, frame)
#     cnt += 1
#     success, frame = cap.read()

print(cnt)
# cv2.destroyWindow('capture')
cap.release()