import cv2
import glob
import time
import numpy as np

# images = glob.glob('../LoFTR/assets/odometry/09/result/*.png')
images = sorted(glob.glob('../LoFTR/assets/odometry/09/masked/*.png'))


fps = 20
# size = (729, 108)
size = (1226, 370)
# vWriter = cv2.VideoWriter('../loftr.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
vWriter = cv2.VideoWriter('../ld.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

# for i in range (len(images)): # 152-636
for i in range (60, 297): # 152-636
    path = images[i]
    frame = cv2.imread(path, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    vWriter.write(frame)
    time.sleep(0.005)
vWriter.release()
cv2.destroyAllWindows()

