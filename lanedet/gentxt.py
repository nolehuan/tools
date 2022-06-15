# generate txt file

# file = open('../FCLane/FCLane_train_gt.txt', 'w')
# for i in range(434): # 820
#     pic_path = "group0/" + str(3 * i).zfill(6) + ".jpg laneseg_label/group0/" + str(3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(128):
#     pic_path = "group1/" + str(10000 + 3 * i).zfill(6) + ".jpg laneseg_label/group1/" + str(10000 + 3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(76):
#     pic_path = "group2/" + str(20000 + 3 * i).zfill(6) + ".jpg laneseg_label/group2/" + str(20000 + 3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# file.close()

# file = open('../FCLane/FCLane_train.txt', 'w')
# for i in range(434): # 820
#     pic_path = "group0/" + str(3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(128):
#     pic_path = "group1/" + str(10000 + 3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(76):
#     pic_path = "group2/" + str(20000 + 3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# file.close()

# file = open('./eigen_test_files_with_gt.txt', 'w')
# for i in range(297):
#     img_path = "2011_09_26_drive_0015/2011_09_26/2011_09_26_drive_0015_sync/image_02/data/" + str(i).zfill(10) + ".png None 721.5377"
#     file.write(img_path)
#     file.write("\n")

import glob
import numpy as np
import cv2

file = open('../LoFTR/assets/odometry/09/kitti_train_gt.txt', 'w')

label_files = '../LoFTR/assets/odometry/09/json/*/label.png'
for label_path in glob.glob(label_files):
    exists = [0, 0, 0, 0]
    png_name = label_path.split('\\')[-2][:-5]
    png = cv2.imread(label_path, 0)

    png = np.asarray(png)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(png)
    while maxVal > 4:
        if maxVal == 113: # yellow
            png[np.where(png == maxVal)] = 3
            exists[2] = 1
            _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

        elif maxVal == 75: # green
            png[np.where(png == maxVal)] = 2
            exists[1] = 1
            _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

        elif maxVal == 38: # red
            png[np.where(png == maxVal)] = 1
            exists[0] = 1
            _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

        elif maxVal == 14: # blue
            png[np.where(png == maxVal)] = 4
            exists[3] = 1
            _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

        else:
            png[np.where(png == maxVal)] = 0
            _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

    pic_path = "/image_2/" + png_name + ".png /label/" + png_name + ".png" \
                + ' ' + str(exists[0]) + ' ' + str(exists[1]) + ' ' + str(exists[2]) + ' ' + str(exists[3])
    file.write(pic_path)
    file.write("\n")


