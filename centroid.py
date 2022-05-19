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

    points11 = points1[:2500, :] # x
    points11 = points11[points11[:, 0].argsort()]

    points12 = points1[2500:, :] # z

    points2 = points[5168:, :]
    indice2 = points2[:, 0].argsort()
    points2 = points2[indice2] # sort by x

    n1 = points1.shape[0] # 5168
    n2 = points2.shape[0] # 863
    n11 = points11.shape[0]
    n12 = points12.shape[0]

    x_min11 = points11[0, 0]
    x_max11 = points11[-1, 0]
    centroid11 = []
    nc11 = int((x_max11 - x_min11) / 2)
    start = 0
    end = 0
    for i in range(nc11):
        start = end
        while (points11[end, 0] <= (x_min11 + i * 2 + 2)): end += 1
        if start == end: continue
        segment11 = points11[start : end, :]
        mean11 = np.mean(segment11, axis=0)
        centroid11.append(mean11)
    centroid11 = np.array(centroid11)

    z_min12 = points12[0, 2]
    z_max12 = points12[-1, 2]
    centroid12 = []
    nc12 = int((z_max12 - z_min12) / 2)
    start = 0
    end = 0
    for i in range(nc12):
        start = end
        while (points12[end, 2] <= (z_min12 + i * 2 + 2)): end += 1
        if start == end: continue
        segment12 = points12[start : end, :]
        mean12 = np.mean(segment12, axis=0)
        centroid12.append(mean12)
    centroid12 = np.array(centroid12)

    x_min2 = points2[0, 0]
    x_max2 = points2[-1, 0]
    centroid2 = []
    nc2 = int((x_max2 - x_min2) / 2)
    start = 0
    end = 0
    for i in range(nc2):
        start = end
        while (points2[end, 0] <= (x_min2 + i * 2 + 2)): end += 1
        if start == end: continue
        segment2 = points2[start : end, :]
        mean2 = np.mean(segment2, axis=0)
        centroid2.append(mean2)
    centroid2 = np.array(centroid2)

    '''
    dist_thresh = 2
    centroid1 = [np.mean(points1[0 : 20, :], axis=0)]
    start = 20
    end = 21
    while end < n1:
        segment = points1[start : end, :]
        mean = np.mean(segment, axis=0)
        diff = centroid1[-1] - mean
        dist = np.linalg.norm(diff)
        if dist > dist_thresh:
            centroid1.append(mean)
            start = end
        end += 1
    centroid1 = np.array(centroid1)

    centroid2 = [np.mean(points2[0 : 20, :], axis=0)]
    start = 20
    end = 21
    while end < n2:
        segment = points2[start : end, :]
        mean = np.mean(segment, axis=0)
        diff = centroid2[-1] - mean
        dist = np.linalg.norm(diff)
        if dist > dist_thresh:
            centroid2.append(mean)
            start = end
        end += 1
    centroid2 = np.array(centroid2)
    '''

    '''
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
    '''

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    from sim3 import read_gt
    x_path = []
    y_path = []
    z_path = []
    odometry = [0]
    read_gt(x_path, y_path, z_path, odometry)
    x_path1 = x_path[:410]
    y_path1 = y_path[:410]
    z_path1 = z_path[:410]
    x_path2 = x_path[410:]
    y_path2 = y_path[410:]
    z_path2 = z_path[410:]

    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    # ax.plot3D(x, y, z, 'b.')
    # ax.plot3D(points11[:, 0], points11[:, 1], points11[:, 2], 'b.')
    ax.plot3D(centroid11[:, 0], centroid11[:, 1], centroid11[:, 2], 'm.')
    # ax.plot3D(points12[:, 0], points12[:, 1], points12[:, 2], 'b.')
    ax.plot3D(centroid12[:, 0], centroid12[:, 1], centroid12[:, 2], 'm.')
    # ax.plot3D(centroid1[:, 0], centroid1[:, 1], centroid1[:, 2], 'y.')
    # ax.plot3D(points2[:, 0], points2[:, 1], points2[:, 2], 'b.')
    ax.plot3D(centroid2[:, 0], centroid2[:, 1], centroid2[:, 2], 'm.')
    # ax.plot3D(x_path1, y_path1, z_path1, 'r.')
    # ax.plot3D(x_path2, y_path2, z_path2, 'c.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-150, 100)
    ax.set_ylim3d(-80, 30)
    ax.set_zlim3d(200, 450)
    plt.show()

with open("./files/centroid.txt", "w") as f:
    i = 0
    for i in range (centroid11.shape[0]):
        s = str(centroid11[i, 0]) + ' ' + str(centroid11[i, 1]) + ' ' + str(centroid11[i, 2]) + '\n'
        f.write(s)
    i = 0
    for i in range (centroid12.shape[0]):
        s = str(centroid12[i, 0]) + ' ' + str(centroid12[i, 1]) + ' ' + str(centroid12[i, 2]) + '\n'
        f.write(s)
    i = 0
    for i in range (centroid2.shape[0]):
        s = str(centroid2[i, 0]) + ' ' + str(centroid2[i, 1]) + ' ' + str(centroid2[i, 2]) + '\n'
        f.write(s)
    f.close()
