import numpy as np
import cv2

if __name__ == "__main__":
    path = "./files/0000000000.png" # 1242 375
    img = cv2.imread(path, -1)

    y = 195
    while y < 375:
        cv2.line(img, (0, y), (1242, y), (255, 144, 30), 1, cv2.LINE_AA)
        y += 18
    x = 3
    while x < 1242:
        cv2.line(img, (x, 195), (x, 375), (255, 144, 30), 1, cv2.LINE_AA)
        x += 12
    
    nx = 38
    ny = 0
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 33
    ny = 1
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 28
    ny = 2
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 23
    ny = 3
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 17
    ny = 4
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 11
    ny = 5
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 5
    ny = 6
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)
    nx = 0
    ny = 7
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 0, 0), cv2.LINE_AA)

    nx = 46
    ny = 0
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 44
    ny = 1
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 42
    ny = 2
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 40
    ny = 3
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 38
    ny = 4
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 36
    ny = 5
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 34
    ny = 6
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 32
    ny = 7
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 30
    ny = 8
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)
    nx = 28
    ny = 9
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (0, 255, 0), cv2.LINE_AA)

    nx = 53
    ny = 0
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 55
    ny = 1
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 57
    ny = 2
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 58
    ny = 3
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 60
    ny = 4
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 62
    ny = 5
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 64
    ny = 6
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 66
    ny = 7
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 68
    ny = 8
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)
    nx = 70
    ny = 9
    points = np.array([[3+12*nx, 195+ny*18], [15+12*nx, 195+ny*18], [15+12*nx, 213+ny*18], [3+12*nx, 213+ny*18]])
    cv2.fillConvexPoly(img, points, (255, 255, 0), cv2.LINE_AA)

    cv2.imshow("demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("./files/grid_fill.png", img)

