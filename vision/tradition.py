from copy import deepcopy
import cv2
import numpy as np

def watershed_algorithm(image):

    blur = cv2.pyrMeanShiftFiltering(image, sp=10, sr=100)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    dilate = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow("dilate", dilate)
    cv2.waitKey(0)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, maskSize=3)
    dist_out = deepcopy(dist)
    cv2.normalize(dist, dist_out, 0, 1.0, cv2.NORM_MINMAX)
    _, surface = cv2.threshold(dist_out, dist_out.max() * 0.6, 255, cv2.THRESH_BINARY)
    cv2.imshow("surface", surface)
    cv2.waitKey(0)

    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(dilate, surface_fg)
    _, markers = cv2.connectedComponents(surface_fg)
    print(markers)

    markers += 1 # Add one to all labels so that sure background is not 0, but 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    cv2.imshow("result", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    image_path = "./files/money.jpg"
    image = cv2.imread(image_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    watershed_algorithm(image)
