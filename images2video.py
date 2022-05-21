import cv2
import glob
import time

images = glob.glob('../LoFTR/assets/odometry/09/result/*.png')
idx = [float(image[-8 : -4]) for image in images]

# images = sorted(glob.glob('../LoFTR/assets/odometry/09/masked/*.png'))

fps = 20
size = (729, 108)
# fps = 30
# size = (1226, 370)
vWriter = cv2.VideoWriter('../loftr.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
# vWriter = cv2.VideoWriter('../ld.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
for i in range (18, 278): # 152-636
# for i in range (60, 297): # 152-636
    frame = cv2.imread(images[i], -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    vWriter.write(frame)
    time.sleep(0.005)
vWriter.release()
cv2.destroyAllWindows()

