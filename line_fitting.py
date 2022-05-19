import numpy as np
from numpy.core.fromnumeric import shape
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


def readRt(Rts):
    Rt_path = './odom09.txt'
    with open(Rt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            Rt = line.split(' ')
            if len(Rt) == 13: Rts.append(Rt)
    f.close()
    return
# Rts = []
# readRt(Rts)
# Rts = np.array(Rts)
# ts = Rts[:, 10 : 12]
# max_t = max(ts[:, 2])
# min_t = min(ts[:, 2])

'''
sim3 = np.load('./files/kitti_09_ape/alignment_transformation_sim3.npy')
path = './files/TriangulatedPoint3d_lane1.txt'
out_file = './files/lane1.txt'
with open(path, 'r') as f, open(out_file, 'w') as outf:
    for line in f.readlines():
        line = line.strip()
        point = line.split(' ')
        if len(point) == 3:
            tx = float(point[0])
            ty = float(point[1])
            tz = float(point[2])
            t = np.array([tx, ty, tz]).reshape([3, 1])
            R = sim3[:3, :3].reshape([3, 3])
            dt = sim3[:3, 3].reshape([3, 1])
            t = np.dot(R, t) + dt
            for i in range (3):
                outf.write(str(round(t[i][0], 6)))
                if i != 2: outf.write(' ')
            outf.write('\n')
f.close()
outf.close()
'''

points = []
path = './files/lane1_filtered3.txt'
# path = './files/TriangulatedPoint3d_lane1.txt'
with open(path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        point = line.split(' ')
        if len(point) == 3: points.append(point)
f.close()
points = np.array(points).astype('float64')
# points = points[points[:, 2].argsort()] # sort by col
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]
'''
from sim3 import read_gt
x_path = []
y_path = []
z_path = []
odometry = [0]
read_gt(x_path, y_path, z_path, odometry)
from sim3 import readVIO
pos = []
readVIO(pos)
pos = np.array(pos).reshape([len(pos), 3])
xVIO = pos[:, 0]
yVIO = pos[:, 1]
zVIO = pos[:, 2]
xVIO = xVIO.tolist()
yVIO = yVIO.tolist()
zVIO = zVIO.tolist()


fig = plt.figure()
ax = plt.gca(projection = '3d')
ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b.')
ax.plot3D(x_path, y_path, z_path, 'g.')
ax.plot3D(xVIO, yVIO, zVIO, 'y.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-150, 100)
ax.set_ylim3d(-150, 100)
ax.set_zlim3d(200, 450)
plt.show()
'''

inliers = points

'''
# estimator = DBSCAN(eps=0.3, min_samples=40)
# estimator = DBSCAN(eps=0.29, min_samples=8)
estimator = DBSCAN(eps=50, min_samples=5)
estimator.fit(inliers)
label_pred = estimator.labels_
maxlabel = max(estimator.labels_)

# outliers = np.vstack((outliers, inliers[label_pred != maxlabel]))
outliers = inliers[label_pred != maxlabel]
inliers = inliers[label_pred == maxlabel]

# inliers = inliers[inliers[:, 0].argsort()] # sort by col
'''

x = inliers[:, 0]
y = inliers[:, 1]
z = inliers[:, 2]

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

'''
u = np.arange(len(x))
s = 0.02 * len(u)
spx = UnivariateSpline(u, x, s = s)
# spx.set_smoothing_factor(0.6)
spy = UnivariateSpline(u, y, s = s)
# spy.set_smoothing_factor(0.6)
spz = UnivariateSpline(u, z, s = s)
# spz.set_smoothing_factor(0.6)
x_new = spx(u)
y_new = spy(u)
z_new = spz(u)
'''

# x-y x-z
'''
r = np.arange(len(x))
R = r.reshape(-1, 1)
R = np.hstack([np.ones_like(R), R, R ** 2, R ** 3])
ransac_x = linear_model.RANSACRegressor()
ransac_x.fit(R, x)
line_x_ransac = ransac_x.predict(R)
ransac_y = linear_model.RANSACRegressor()
ransac_y.fit(R, y)
line_y_ransac = ransac_y.predict(R)
ransac_z = linear_model.RANSACRegressor()
ransac_z.fit(R, z)
line_z_ransac = ransac_z.predict(R)
'''
'''
x = np.sort(x)
X = x.reshape(-1, 1)
X = np.hstack([np.ones_like(X), X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8])
ransac_y = linear_model.RANSACRegressor()
ransac_y.fit(X, y)
line_y_ransac = ransac_y.predict(X)

ransac_z = linear_model.RANSACRegressor()
ransac_z.fit(X, z)
line_z_ransac = ransac_z.predict(X)

# 2D
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(x, y, 'b.')
ax.plot(x, line_y_ransac, 'g.')
ax = fig.add_subplot(212)
ax.plot(x, z, 'b.')
ax.plot(x, line_z_ransac, 'g.')
plt.show()
'''


# '''
# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot3D(x, y, z, 'b.')
# ax.plot3D(line_x_ransac, line_y_ransac, line_z_ransac, 'r.')
# ax.plot3D(x, line_y_ransac, line_z_ransac, 'g.')
# ax.plot3D(x_new, y_new, z_new, 'g.')
# ax.plot(x_new, y_new, z_new, label='curve', c='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-150, 100)
ax.set_ylim3d(-150, 100)
ax.set_zlim3d(200, 450)
plt.show()
# '''

# with open("./processed_lane0.txt", "w") as f:
#     for i in range (x_new.shape[0]):
#         # s = str(line_x_ransac[i]) + ' ' + str(line_y_ransac[i]) + ' ' + str(zz[i]) + '\n'
#         s = str(x_new[i]) + ' ' + str(y_new[i]) + ' ' + str(z_new[i]) + '\n'
#         f.write(s)
#     f.write('\n')
#     for i in range (inliers.shape[0]):
#         s = str(inliers[i][0]) + ' ' + str(inliers[i][1]) + ' ' + str(inliers[i][2]) + '\n'
#         f.write(s)
#     f.write('\n')
#     for i in range (outliers.shape[0]):
#         s = str(outliers[i][0]) + ' ' + str(outliers[i][1]) + ' ' + str(outliers[i][2]) + '\n'
#         f.write(s)
#     f.close()
