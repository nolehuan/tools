import cv2

if __name__ == "__main__":
    image = cv2.imread("", -1)
    blur = cv2.pyrMeanShiftFiltering(image, sp=10, sr=100)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow("bg", bg)
    cv2.waitKey(0)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, maskSize=3)
    dist = cv2.normalize(dist, alpha=0, beta=1, normType=cv2.NORM_MINMAX)
