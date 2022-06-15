import numpy as np
import cv2
import matplotlib.pyplot as plt


def perspective_transform(img, transform_matrix, transform_size = (1, 1)):
    if transform_size == (1, 1):
        transform_size = (img.shape[1], img.shape[0])
    perspective_img = cv2.warpPerspective(img, transform_matrix,
                                            transform_size,
                                            flags=cv2.INTER_LINEAR)
    return perspective_img


def get_perspective_mat(src_points, target_points):
    transform_matrix = cv2.getPerspectiveTransform(src_points, target_points)
    return transform_matrix

img_path = './0000000001.png'
img = cv2.imread(img_path, -1)

src_points = np.array([[512., 192.], [424., 224.], [712., 248.], [464., 288.]], dtype="float32")
target_points = np.array([[508., 192.], [393., 231.], [746., 270.], [393., 345.]], dtype="float32")

transform_matrix = get_perspective_mat(src_points, target_points)

perspective_img = perspective_transform(img, transform_matrix, transform_size = (1, 1))

cv2.imshow("img", perspective_img)
cv2.waitKey(0)
