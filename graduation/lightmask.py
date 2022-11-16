import cv2

img = cv2.imread("../lightmask.png")
cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("demo", 1920, 1080)
cv2.setWindowProperty("demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("demo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
