import cv2
import numpy as np

def watershed_algorithm(image_path):
    image = cv2.imread(image_path, -1)
    blur = cv2.pyrMeanShiftFiltering(image, sp=10, sr=100)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    # 形态学操作 距离变换 寻找种子
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow("sure_bg", sure_bg)
    cv2.waitKey(0)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, maskSize=3)
    dist = cv2.normalize(dist, alpha=0, beta=1, normType=cv2.NORM_MINMAX)
    _, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)
    cv2.imshow("surface", surface)
    cv2.waitKey(0)

    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(sure_bg, surface_fg)
    _, markers = cv2.connectedComponents(surface_fg)
    print(markers)

    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    cv2.imshow("result", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    watershed_algorithm("./.png")
