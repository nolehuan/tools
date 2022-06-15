# post processing of labelme_json_to_dataset generated png files

import glob
import cv2
import numpy as np

'''
path = '../ICPS_RoadSurfaceClassification_202201v3/svm.png'
png = cv2.imread(path, -1)
# png = cv2.resize(png, (500, 440))
png = png[20:450, 115:615]
cv2.imshow(".", png)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path, png)
'''

png_file = '../LoFTR/assets/kitti/label/0000000000.png'
png = cv2.imread(png_file, 0)
_, png = cv2.threshold(png, 0.5, 255, cv2.THRESH_BINARY)
cv2.imshow(".", png)
cv2.waitKey(0)

png_files = '../FCLane/group2/*/label.png'
for pngfile in glob.glob(png_files):
    # png_name = pngfile.split('\\')[-1][:-4]
    png_name = pngfile.split('\\')[-2][:-5]
    png = cv2.imread(pngfile, 0) # 113 75 38 14

    png = np.asarray(png)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(png)
    print(maxVal)
    print(maxLoc)
    while maxVal > 0:
        png[np.where(png == maxVal)] = 0
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(png)
        print(maxVal)
        print(maxLoc)

    # i = 1
    # while maxVal > i:
    #     png[np.where(png == maxVal)] = i
    #     _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #     print("i = ",i)
    #     print(maxVal)
    #     print(maxLoc)
    #     i += 1

    # while maxVal > 4:
    #     if maxVal == 113: # yellow
    #         png[np.where(png == maxVal)] = 2 # yellow_dash 1
    #         _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #         # print(maxVal)
    #         # print(maxLoc)
            
    #     elif maxVal == 75: # green
    #         png[np.where(png == maxVal)] = 4 # white_solid 2
    #         _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #         # print(maxVal)
    #         # print(maxLoc)
            
    #     elif maxVal == 38: # red
    #         png[np.where(png == maxVal)] = 3 # road_curb 3
    #         _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #         # print(maxVal)
    #         # print(maxLoc)
            
    #     elif maxVal == 14: # blue
    #         png[np.where(png == maxVal)] = 1 # white_dash 4
    #         _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #         # print(maxVal)
    #         # print(maxLoc)
            
    #     else:
    #         png[np.where(png == maxVal)] = 0
    #         _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
    #         # print(maxVal)
    #         # print(maxLoc)


    # save_path = "D:/Users/nole/Desktop/FCLane/laneseg_label/group0/" + png_name + ".png"
    # cv2.imwrite(save_path, png)

