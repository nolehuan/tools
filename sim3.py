import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
# sim3 = np.load('./files/kitti_09_ape_part/alignment_transformation_sim3.npy')
sim3 = np.array([[4.11801, 0.52509, 10.6321, -182.676],
                [-1.6277, -11.2366, 1.18151, -12.4243],
                [10.5214, -1.93319, -3.97967, 83.7326],
                [      0,        0,       0,        1]])

def read_vio(points):
    pts_file = './files/points.txt'
    with open(pts_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            xyz = line.split(' ')
            if len(xyz) == 3:
                points.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    f.close()

def readVIO(pos):
    traj_file = './files/KeyFrameTrajectory09_part.txt'
    with open(traj_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 9:
                tx = float(pose[2])
                ty = float(pose[3])
                tz = float(pose[4])
                t = np.array([tx, ty, tz]).reshape([3, 1])

                #'''
                R = sim3[:3, :3].reshape([3, 3])
                dt = sim3[:3, 3].reshape([3, 1])
                t = np.dot(R, t) + dt
                # s = np.linalg.det(R) ** (1/3)
                # t = s * np.dot(R / s, t) + dt
                pos.append(np.array(t))
                #'''
                '''
                qx = float(pose[5])
                qy = float(pose[6])
                qz = float(pose[7])
                qw = float(pose[8])
                q = [qx, qy, qz, qw]
                # print(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)
                R = Rotation.from_quat(q)
                R = R.as_matrix()
                # q_ = Rotation.from_matrix(R)
                # q_ = q_.as_quat()
                # print(np.transpose(R))
                # print(np.dot(np.transpose(R), R))
                Rt = np.hstack([R, t])
                a = np.array([0, 0, 0, 1])
                T = np.vstack([Rt, a])
                Tsim3 = np.dot(sim3, T)
                pos.append(Tsim3[:3, 3])
                '''
                ''' incorrect
                qx = float(pose[5])
                qy = float(pose[6])
                qz = float(pose[7])
                qw = float(pose[8])
                q = [qx, qy, qz, qw]
                R = Rotation.from_quat(q)
                R = R.as_matrix()
                t = - np.dot(np.transpose(R), t)

                sR = sim3[:3, :3].reshape([3, 3])
                dt = sim3[:3, 3].reshape([3, 1])
                t = np.dot(sR, t) + dt
                pos.append(np.array(t))
                '''

    f.close()

def read_gt(x_path, y_path, z_path, odometry):
    file_path = './files/kitti_09_gt_part.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pose = line.split(' ')
            if len(pose) == 8:
                tx = float(pose[1])
                ty = float(pose[2])
                tz = float(pose[3])
                x_path.append(tx)
                y_path.append(ty)
                z_path.append(tz)
                if len(x_path) > 1:
                    odometry.append(odometry[-1] + math.sqrt((tx - x_path[-2])**2 + (ty - y_path[-2])**2 + (tz - z_path[-2])**2))
    f.close()


if __name__ == '__main__':
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    read_gt(x_path, y_path, z_path, odometry)

    pos = []
    readVIO(pos)
    pos = np.array(pos).reshape([len(pos), 3])
    xVIO = pos[:, 0]
    yVIO = pos[:, 1]
    zVIO = pos[:, 2]
    xVIO = xVIO.tolist()
    yVIO = yVIO.tolist()
    zVIO = zVIO.tolist()

    VIO_odom = [0]
    for i in range (1, pos.shape[0]):
        VIO_odom.append(VIO_odom[-1] + math.sqrt((xVIO[i] - xVIO[i-1])**2 + (yVIO[i] - yVIO[i-1])**2 + (zVIO[i] - zVIO[i-1])**2))

    # points = []
    # read_vio(points)
    # points = np.array(points)
    # x_vio = points[:, 0]
    # y_vio = points[:, 1]
    # z_vio = points[:, 2]
    # x_vio = x_vio.tolist()
    # y_vio = y_vio.tolist()
    # z_vio = z_vio.tolist()

    # fig = plt.figure('path')
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_path, y_path, z_path, label='gnss', c='g')
    # ax.plot(x_vio, y_vio, z_vio, label='vio', c='b')
    # ax.set_xlabel('E')
    # ax.set_ylabel('N')
    # ax.set_zlabel('U')

    # # ax.set_xlim(-200, 200)
    # # ax.set_ylim(-40, 40)
    # # ax.set_zlim(0, 600)
    # ax.legend()
    # plt.show()

    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot3D(x_path, y_path, z_path, 'b.')
    # ax.plot3D(x_vio, y_vio, z_vio, 'g*')
    ax.plot3D(xVIO, yVIO, zVIO, 'y.')
    ax.set_xlabel('E')
    ax.set_ylabel('U')
    ax.set_zlabel('N')
    ax.set_xlim3d(-150, 100)
    ax.set_ylim3d(-150, 100)
    ax.set_zlim3d(200, 450)
    plt.show()
