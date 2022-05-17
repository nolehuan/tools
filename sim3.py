import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
import pandas as pd

sim3 = np.load('./kitti_09_ape/alignment_transformation_sim3.npy')


def readVIO(pos):
    traj_file = './KeyFrameTrajectory09.txt'
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

# pos = []
# readVIO(pos)
# pos = np.array(pos)


# df = pd.DataFrame(pos)
# df.rename(columns = {0: "x", 1: "y", 2: "z"}, inplace = True)
# writer = pd.ExcelWriter("./vo.xlsx")
# df.to_excel(writer)
# writer.save()
# # df.to_csv("./kitti_path09.csv")

# odom = [0]
# for i in range (1, pos.shape[0], 1):
#     odom.append(odom[-1] + math.sqrt((pos[i, 0] - pos[i-1, 0])**2 + (pos[i, 1] - pos[i-1, 1])**2 + (pos[i, 2] - pos[i-1, 2])**2))

def readgt(pos):
    traj_file = './kitti_09_gt_vio.txt'
    with open(traj_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 8:
                tx = float(pose[1])
                ty = float(pose[2])
                tz = float(pose[3])
                t = np.array([tx, ty, tz])

                pos.append(t)
    f.close()


pos_gt = []
odometry = [0]
readgt(pos_gt)
pos_gt = np.array(pos_gt)
pos_gt = pos_gt - pos_gt[0]
pos_gt[:, 1] = -pos_gt[:, 1]

df = pd.DataFrame(pos_gt)
df.rename(columns = {0: "x", 1: "y", 2: "z"}, inplace = True)
writer = pd.ExcelWriter("./gnss.xlsx")
df.to_excel(writer)
writer.save()
# df.to_csv("./kitti_path09.csv")

for i in range (1, pos_gt.shape[0], 1):
    odometry.append(odometry[-1] + math.sqrt((pos_gt[i, 0] - pos_gt[i-1, 0])**2 + (pos_gt[i, 1] - pos_gt[i-1, 1])**2 + (pos_gt[i, 2] - pos_gt[i-1, 2])**2))


# from kitti_path_player import readGNSS
# x_path = []
# y_path = []
# z_path = []
# odometry = [0]

# readGNSS(x_path, y_path, z_path, odometry)


# fig = plt.figure('path')
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='vo path', c='g')
# ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label='gnss path', c='r')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim(-200, 200)
# ax.set_ylim(-40, 40)
# ax.set_zlim(0, 600)
# ax.legend()
# plt.show()

fig = plt.figure('odom')
ax = fig.add_subplot(111)
# ax.plot(odom, -pos[:, 1], label='vo odom', c='g')
ax.plot(odometry, -pos_gt[:, 1], label='gnss odom', c='r')
ax.set_xlim(-10, 600)
ax.set_ylim(0, 40)
ax.legend()
plt.show()

