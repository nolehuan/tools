import glob
import cv2
import numpy as np
from tqdm import tqdm

png_files = '../FCLane/group2/*/label.png'
file = open('../FCLane/FCLane_train_gt.txt', 'a', encoding='utf-8')

with tqdm(total = 76) as pbar:
    for pngfile in glob.glob(png_files):
        exists = [0, 0, 0, 0]
        png_name = pngfile.split('\\')[-2][:-5]
        png = cv2.imread(pngfile, 0)

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

        save_path = "D:/Users/buaa/Desktop/FCLane/laneseg_label/culane_style/group2/" + png_name + ".png"
        cv2.imwrite(save_path, png)

        pic_path = "/group2/" + png_name + ".jpg /laneseg_label/culane_style/group2/" + png_name + ".png" \
                    + ' ' + str(exists[0]) + ' ' + str(exists[1]) + ' ' + str(exists[2]) + ' ' + str(exists[3])
        file.write(pic_path)
        file.write("\n")
        pbar.update(1)
file.close()
