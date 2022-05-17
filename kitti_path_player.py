import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gps2xyz import gps_to_ecef, ecef_to_enu

# traj_file = './KeyFrameTrajectory09.txt'
# poses = []
# with open(traj_file, 'r') as f:
#     for line in f.readlines():
#         line = line.strip()
#         pose = line.split(' ')
#         if len(pose) == 9:
#             poses.append(pose[1:])
# f.close()
# file_out = './Traj09.txt'
# with open(file_out, 'w') as f:
#     for pose in poses:
#         for i in range (8):
#             f.write(str(pose[i]))
#             if i != 7 : f.write(' ')
#         f.write('\n')
# f.close()

def readVIO(poses):
    traj_file = './KeyFrameTrajectory09.txt'
    with open(traj_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 9:
                tx = float(pose[2])
                ty = float(pose[3])
                tz = float(pose[4])
                t = [tx, ty, tz]
                poses.append(t)
    f.close()

def readGNSS(x_path, y_path, z_path, odometry):
    # file_path = '../dataset/KITTI/raw/residential/2011_09_30_drive_0033/2011_09_30/2011_09_30_drive_0033_sync/oxts/data/*'
    file_path = './data/*'
    gps_files = sorted(glob.glob(file_path))
    gps_ref = np.loadtxt(gps_files[0])
    gps_ref_lla = gps_ref[0:3]

    # for i in range(0, len(gps_files)):
    for i in range(15, 1155):
        gps_cur = np.loadtxt(gps_files[i])
        gps_cur_lla = gps_cur[0:3]
        x, y, z = gps_to_ecef(gps_cur[0], gps_cur[1], gps_cur[2])
        x_enu, y_enu, z_enu = ecef_to_enu(x, y, z, gps_ref_lla[0], gps_ref_lla[1], gps_ref_lla[2])

        if i > 15:
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
    poses = []
    readVIO(poses)
    poses = np.array(poses)
    poses = np.multiply(poses, 21.9887)
    x_vio = poses[:,0]
    y_vio = poses[:,1]
    z_vio = poses[:,2]


    fig = plt.figure('path')
    ax = fig.add_subplot(211, projection='3d')
    ax.plot(x_path, y_path, z_path, label='kitti path', c='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(0, 500)
    # ax.set_ylim(0, 500)
    # ax.set_zlim(0, 500)
    ax = fig.add_subplot(212)
    ax.plot(odometry, z_path, label='odom path', c='r')
    
    plt.show()

    # data = [odometry, z_path]
    # data = np.array(data)
    # data = np.transpose(data)
    # df = pd.DataFrame(data)
    # # df = pd.DataFrame(odometry)
    # # df = df.append(pd.DataFrame(z_path), ignore_index=True)
    # df.rename(columns = {0: "odom", 1: "height"}, inplace = True)
    # writer = pd.ExcelWriter("./kitti_path09.xlsx")
    # df.to_excel(writer)
    # writer.save()
    # # df.to_csv("./kitti_path09.csv")

