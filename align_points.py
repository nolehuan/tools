from ctypes import pointer
import numpy as np
from centroid import read_pts
from sim3 import read_gt

if __name__ == "__main__":
    points = []
    path = "./files/centroid_split_fit.txt"
    read_pts(points, path)
    points = np.array(points)
    
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    file_path = './files/kitti_09_gt_part_align.txt'
    read_gt(file_path, x_path, y_path, z_path, odometry)

    align_points = [points[0, :]]
    cur = 1
    idx = 1
    while cur < points.shape[0]:
        cur_point = points[cur, :]
        dist = odometry[idx] - odometry[idx - 1]
        cur += 1