import glob, json
import cv2
import numpy as np
from tqdm import tqdm

# hightest_points = {"frame": "*.jpg", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}

png_files = '../dataset/FCLane/group2/*/label.png'
min_y = 280

with tqdm(total = 76) as pbar:
    for pngfile in glob.glob(png_files):
        hightest_points = {"frame": "*.jpg", "points": [[],[],[],[]]}
        png_name = pngfile.split('\\')[-2][:-5]
        hightest_points['frame'] = png_name + ".jpg"
        png = cv2.imread(pngfile, 0)

        png = np.asarray(png)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
        while maxVal > 0:
            if maxLoc[1] < min_y:
                min_y = maxLoc[1]
            if maxVal == 113: # yellow
                png[np.where(png == maxVal)] = 0
                hightest_points['points'][2] = maxLoc
                _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

            elif maxVal == 75: # green
                png[np.where(png == maxVal)] = 0
                hightest_points['points'][1] = maxLoc
                _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

            elif maxVal == 38: # red
                png[np.where(png == maxVal)] = 0
                hightest_points['points'][0] = maxLoc
                _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

            elif maxVal == 14: # blue
                png[np.where(png == maxVal)] = 0
                hightest_points['points'][3] = maxLoc
                _, maxVal, _, maxLoc = cv2.minMaxLoc(png)

            else:
                png[np.where(png == maxVal)] = 0
                _, maxVal, _, maxLoc = cv2.minMaxLoc(png)
        
        with open("../dataset/FCLane/hightest_points.json", 'a', encoding='utf-8') as f:
            json.dump(hightest_points, f)
            f.write("\n")
            f.close()

        pbar.update(1)
print(min_y)
