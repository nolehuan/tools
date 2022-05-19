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
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot3D(x, y, z, 'b.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-150, 100)
    ax.set_ylim3d(-150, 100)
    ax.set_zlim3d(200, 450)
    plt.show()
