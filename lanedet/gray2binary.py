# 灰度图转二值图

import cv2
import numpy as np

'''
image = cv2.imread("../LoFTR/assets/odometry/09/label/000016.png", -1)
image[np.where(image == 2)] = 255
cv2.imshow("demo", image)
cv2.waitKey(0)
'''


# img = cv2.imread(r"../dataset/culane/laneseg_label_w16/driver_23_30frame/05151640_0419.MP4/00000.png", 0)
# # img = cv2.imread(r"../FCLane/laneseg_label/group1/010000.png")
# # img = cv2.imread(r"../FCLane/group1/010000_json/label.png")
# img = cv2.imread(r"../FCLane/template/temp_json/label.png", 0)

# ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
# cv2.namedWindow('capture')
# cv2.imshow('capture', thresh)
# print(img.size)
# print(thresh.size)
# cv2.waitKey(0)
# cv2.destroyWindow('capture')

# img = cv2.imread(r"D:/Users/nole/Desktop/FCLane/template/temp_json/label.png")
# img = cv2.imread(r"D:/Users/nole/Desktop/dataset/culane/laneseg_label_w16/driver_23_30frame/05151640_0419.MP4/00000.png")
# img = cv2.imread(r"D:/Users/nole/Desktop/dataset/culane/laneseg_label_w16/driver_161_90frame/06030819_0755.MP4/00000.png")
# img = cv2.imread(r"D:/Users/nole/Desktop/dataset/culane/laneseg_label_w16/driver_182_30frame/05312327_0001.MP4/00000.png")
# img = cv2.imread(r"D:/Users/nole/Desktop/dataset/culane/laneseg_label_w16/driver_161_90frame/06031752_0900.MP4/00000.png")
# img = cv2.imread(r"D:/Users/nole/Desktop/FCLane/laneseg_label/group1/010000.png", 0)
# img = cv2.imread(r"D:/Users/buaa/Desktop/FCLane/laneseg_label/culane_style/group2/020000.png", 0)
# img = cv2.imread(r"D:/Users/nole/Desktop/FCLane/laneseg_label/group0/001299.png", 0)
img = cv2.imread(r"D:/Users/buaa/Desktop/LoFTR/assets/odometry/09/label/000068.png", 0)

# _, maxVal, _, maxLoc = cv2.minMaxLoc(img)
# print(maxVal)
# print(maxLoc)
# img = np.asarray(img)

# while maxVal > 0:
#     img[np.where(img == maxVal)] = 0
#     _, maxVal, _, maxLoc = cv2.minMaxLoc(img)
#     # cv2.circle(img, maxLoc, 10, (255, 0, 0))
#     print(maxVal)
#     print(maxLoc)


# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU) # 像素值为1的点变为0
retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
# img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_LINEAR)

# img = img[120:None]

# m = img.max()
# print(m)
cv2.imshow("img", img)
cv2.waitKey(0)
# cv2.imwrite("D:/Users/*/Desktop/*.jpg",img)
cv2.imwrite("../68.png",img)
