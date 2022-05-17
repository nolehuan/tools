import cv2
import glob

images = sorted(glob.glob('../dataset/CULane/driver_23_30frame/05151640_0419.MP4/*.jpg'))

fps = 25
size = (1640, 590)
videoWriter = cv2.VideoWriter('./culane.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
for i in range (len(images)):
    frame = cv2.imread(images[i], -1)
    videoWriter.write(frame)
