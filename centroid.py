import numpy as np
import matplotlib.pyplot as plt

def read_pts(points):
    path = './files/lane1_filtered3.txt'
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            point = line.split(' ')
            if len(point) == 3: points.append(point)
    f.close()

if __name__ == "__main__":
    points = []
    read_pts(points)
    points = np.array(points).astype('float64')
    indice = points[:, 2].argsort()
    points = points[indice] # sort by z

    points1 = points[:5168, :]

    points2 = points[5168:, :]
    indice2 = points2[:, 0].argsort()
    points2 = points2[indice2] # sort by x

    n1 = points1.shape[0] # 5168
    n2 = points2.shape[0] # 863
    centroid1 = []
    centroid2 = []
    for i in range(n1 // 20):
        segment = points1[20 * i : 20 * (i + 1), :]
        centroid = np.mean(segment, axis=0)
        centroid1.append(centroid)
    centroid1 = np.array(centroid1)
    for j in range(n2 // 20):
        segment = points2[20 * j : 20 * (j + 1), :]
        centroid = np.mean(segment, axis=0)
        centroid2.append(centroid)
    centroid2 = np.array(centroid2)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    from sim3 import read_gt
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    read_gt(x_path, y_path, z_path, odometry)

    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot3D(x, y, z, 'b.')
    ax.plot3D(centroid1[:, 0], centroid1[:, 1], centroid1[:, 2], 'yo')
    ax.plot3D(centroid2[:, 0], centroid2[:, 1], centroid2[:, 2], 'go')
    ax.plot3D(x_path, y_path, z_path, 'r.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-150, 100)
    ax.set_ylim3d(-80, 30)
    ax.set_zlim3d(200, 450)
    plt.show()
