from hashlib import new
import matplotlib.pyplot as plt
import numpy as np
from sympy import N
from centroid import read_pts
from sim3 import read_gt

if __name__ == "__main__":
    points = []
    pts_path = "./files/centroid_split_fit.txt"
    read_pts(points, pts_path)
    points = np.array(points).astype('float64')
    
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    gt_path = './files/kitti_09_gt_part_align.txt'
    read_gt(gt_path, x_path, y_path, z_path, odometry)
    # xyz = np.vstack([x_path, y_path, z_path]).reshape([len(x_path), 3])

    align_points = [points[0, :]]
    cur = 1
    idx = 1
    dest_distance = 0
    cur_distance = 0
    while cur < points.shape[0]:
        cur_point = points[cur, :]
        dest_distance = odometry[idx] - odometry[idx - 1]
        cur_distance = np.linalg.norm(cur_point - align_points[-1])
        if cur_distance > dest_distance:
            new_point = align_points[-1] + dest_distance / cur_distance * (cur_point - align_points[-1])
            align_points.append(new_point)
            idx += 1
            continue
        cur += 1
    if len(align_points) < len(x_path):
        new_point = align_points[-1] + dest_distance / cur_distance * (points[-1] - align_points[-1])
        align_points.append(new_point)
    align_points = np.array(align_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    # ax.plot3D(align_points[:, 0], align_points[:, 1], align_points[:, 2], 'c.')
    # ax.plot3D(x_path, y_path, z_path, 'm.')
    ax.scatter(align_points[:, 0], align_points[:, 1], align_points[:, 2], c='c', s=1, label='Ours')
    ax.scatter(x_path, y_path, z_path, c='m', s=1, label='GNSS')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-150, 130)
    ax.set_ylim3d(-80, 30)
    ax.set_zlim3d(200, 450)
    plt.show()

    '''
    timestamps = []
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 8:
                timestamps.append(pose[0])
        f.close()
    timestamps = np.array(timestamps).astype('float64')
    assert timestamps.shape[0] == align_points.shape[0], "error"
    with open("./files/centroid_fit_align.txt", "w") as f:
        i = 0
        for i in range (align_points.shape[0]):
            s = str(round(timestamps[i], 6)) + ' '
            s += str(round(align_points[i, 0], 7)) + ' ' + str(round(align_points[i, 1], 7)) + ' ' + str(round(align_points[i, 2], 7))
            s += ' 1 0 0 0\n'
            f.write(s)
        f.close()
    '''

