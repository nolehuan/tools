import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
import glob
import pandas as pd
from gps2xyz import gps_to_ecef, ecef_to_enu

sim3 = np.load('../tools/files/kitti_09_ape/alignment_transformation_sim3.npy')


def readVIO(pos):
    traj_file = '../tools/files/KeyFrameTrajectory09.txt'
    with open(traj_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 9:
                tx = float(pose[2])
                ty = float(pose[3])
                tz = float(pose[4])
                t = np.array([tx, ty, tz]).reshape([3, 1])
                q1 = float(pose[5])
                q2 = float(pose[6])
                q3 = float(pose[7])
                q4 = float(pose[8])
                q = [q1, q2, q3, q4]
                R = Rotation.from_quat(q)
                R = R.as_matrix()
                # print(np.dot(np.transpose(R), R))
                Rt = np.hstack([R, t])
                a = np.array([0, 0, 0, 1])
                T = np.vstack([Rt, a])
                Tsim3 = np.dot(sim3, T)

                pos.append(Tsim3[:3, 3])
    f.close()

def readGNSS(x_path, y_path, z_path, odometry):
    # file_path = '../dataset/KITTI/raw/residential/2011_09_30_drive_0033/2011_09_30/2011_09_30_drive_0033_sync/oxts/data/*'
    file_path = '../tools/files/data/*'
    gps_files = sorted(glob.glob(file_path))
    gps_ref = np.loadtxt(gps_files[149])
    gps_ref_lla = gps_ref[0:3]

    # for i in range(0, len(gps_files)):
    for i in range(149, 648):
        gps_cur = np.loadtxt(gps_files[i])
        x, y, z = gps_to_ecef(gps_cur[0], gps_cur[1], gps_cur[2])
        x_enu, y_enu, z_enu = ecef_to_enu(x, y, z, gps_ref_lla[0], gps_ref_lla[1], gps_ref_lla[2])

        if i > 149:
            odometry.append(odometry[-1] + math.sqrt((x_enu - x_path[-1])**2 + (y_enu - y_path[-1])**2 + (z_enu - z_path[-1])**2))

        x_path.append(x_enu)
        y_path.append(y_enu)
        z_path.append(z_enu)


if __name__ == '__main__':
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    readGNSS(x_path, y_path, z_path, odometry)
    print(x_path[-1] - x_path[0])
    print(y_path[-1] - y_path[0])
    print(z_path[-1] - z_path[0])

    pos = []
    readVIO(pos)
    pos = np.array(pos)
    x_vio = pos[:, 0]
    y_vio = pos[:, 1]
    z_vio = pos[:, 2]
    print(x_vio[-1] - x_vio[0])
    print(y_vio[-1] - y_vio[0])
    print(z_vio[-1] - z_vio[0])

    vio_odom = [0]
    for i in range (1, pos.shape[0]):
        vio_odom.append(vio_odom[-1] + math.sqrt((x_vio[i] - x_vio[i-1])**2 + (y_vio[i] - y_vio[i-1])**2 + (z_vio[i] - z_vio[i-1])**2))

    fig = plt.figure('path')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_path, y_path, z_path, label='gnss', c='g')
    ax.plot(x_vio, -z_vio, -y_vio, label='vio', c='b')
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_zlabel('U')

    # ax.set_xlim(-200, 200)
    # ax.set_ylim(-40, 40)
    # ax.set_zlim(0, 600)
    ax.legend()
    plt.show()


