from cProfile import label
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import math

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
path = './files/centroid.txt'
# path = './files/lane1_filtered3.txt'
# path = './files/TriangulatedPoint3d_lane1.txt'
from centroid import read_pts
read_pts(points, path)
points = np.array(points).astype('float64')


from sim3 import read_gt
x_path = []
y_path = []
z_path = []
odometry = [0]
file_path = './files/kitti_09_gt_part_align.txt'
read_gt(file_path, x_path, y_path, z_path, odometry)
print(odometry[-1])
# from sim3 import readVIO
# pos = []
# readVIO(pos)
# pos = np.array(pos).reshape([len(pos), 3])
# xVIO = pos[:, 0]
# yVIO = pos[:, 1]
# zVIO = pos[:, 2]
# xVIO = xVIO.tolist()
# yVIO = yVIO.tolist()
# zVIO = zVIO.tolist()

'''
fig = plt.figure()
ax = plt.gca(projection = '3d')
ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b.')
# ax.plot3D(x_path, y_path, z_path, 'g.')
# ax.plot3D(xVIO, yVIO, zVIO, 'y.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-150, 100)
ax.set_ylim3d(-150, 100)
ax.set_zlim3d(200, 450)
plt.show()
'''

'''
inliers = points
# estimator = DBSCAN(eps=0.3, min_samples=40)
# estimator = DBSCAN(eps=0.29, min_samples=8)
estimator = DBSCAN(eps=50, min_samples=5)
estimator.fit(inliers)
label_pred = estimator.labels_
maxlabel = max(estimator.labels_)

# outliers = np.vstack((outliers, inliers[label_pred != maxlabel]))
outliers = inliers[label_pred != maxlabel]
inliers = inliers[label_pred == maxlabel]

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

# segment
x_fit = []
y_fit = []
z_fit = []
# 1-87 == > 1--95
for x in range(-113, 61):
    x_norm = (x + 18.74) / 54.52
    y = -0.2916*(x_norm**5)+0.1924*(x_norm**4)+1.877*(x_norm**3)+0.7459*(x_norm**2)-7.951*x_norm-24.61
    z = 2.881*(x_norm**5)-0.8038*(x_norm**4)-0.9667*(x_norm**3)+11.66*(x_norm**2)+18.85*x_norm+242.2
    x_fit.append(x)
    y_fit.append(y)
    z_fit.append(z)
# 88-190 ==> 70-200
z = 305
# for z in range(306, 525):
while z < 526:
    if z > 515: z += 0.5
    elif z > 500: z += 1
    else: z += 1.5
    z_norm = (z - 402.6) / 79.97
    x = 2.738*(z_norm**7)+2.991*(z_norm**6)-8.286*(z_norm**5)-4.581*(z_norm**4)+17.26*(z_norm**3)-13.99*(z_norm**2)-23.88*z_norm+71.71
    # y = 0.2691*(z_norm**7)+0.4681*(z_norm**6)-1.457*(z_norm**5)-1.487*(z_norm**4)+3.346*(z_norm**3)+1.462*(z_norm**2)-2.919*z_norm-30.79
    y = -0.3637*(z_norm**5)+0.2437*(z_norm**4)+2.132*(z_norm**3)-0.07085*(z_norm**2)-2.589*z_norm-30.59
    x_fit.append(x)
    y_fit.append(y)
    z_fit.append(z)
# 191-208 ==> 184-208
for x in range(67, 100):
    x_norm = (x - 69.68) / 15.3
    z = 0.7733*(x_norm**3)-4.198*(x_norm**2)+3.379*x_norm+529.2
    y = 0.004936*(x_norm**3)+0.07386*(x_norm**2)+1.425*x_norm-28.29
    x_fit.append(x)
    y_fit.append(y)
    z_fit.append(z)

x_fit = np.array(x_fit)
y_fit = np.array(y_fit)
z_fit = np.array(z_fit)

fit_odom = [0]
for i in range(1, z_fit.shape[0]):
    fit_odom.append(fit_odom[-1] + math.sqrt((x_fit[i] - x_fit[i-1])**2 + (y_fit[i] - y_fit[i-1])**2 + (z_fit[i] - z_fit[i-1])**2))
print(fit_odom[-1])
print(z_fit.shape[0])
print(len(odometry))

print(y_path[0])
print(y_fit[0])
y_path = np.array(y_path) - y_path[0]
y_fit = np.array(y_fit) - y_fit[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(odometry, -y_path, label='GNSS', c='c')
ax.plot(fit_odom, -y_fit, label='Ours', c='m')
ax.set_xlim(0, 500)
ax.set_ylim(-10, 60)
ax.set_xlabel('Mileage(m)')
ax.set_ylabel('Elevation(m)')
ax.legend()
plt.show()


# '''
# 3D
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
# ax.plot3D(x, y, z, 'b.')
# ax.plot3D(x_fit, y_fit, z_fit, 'c.')
# ax.plot3D(x_path, y_path, z_path, 'm.')
ax.scatter(x_fit, y_fit, z_fit, c='c', s=1, label='Ours')
ax.scatter(x_path, y_path, z_path, c='m', s=1, label='GNSS')
# ax.plot3D(line_x_ransac, line_y_ransac, line_z_ransac, 'r.')
# ax.plot3D(x, line_y_ransac, line_z_ransac, 'g.')
# ax.plot3D(x_new, y_new, z_new, 'g.')
# ax.plot(x_new, y_new, z_new, label='curve', c='g')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-150, 100)
ax.set_ylim3d(-80, 30)
ax.set_zlim3d(200, 450)
plt.show()
# '''

'''
with open("./files/centroid_split_fit.txt", "w") as f:
    i = 0
    for i in range (x_fit.shape[0]):
        s = str(x_fit[i]) + ' ' + str(y_fit[i]) + ' ' + str(z_fit[i]) + '\n'
        f.write(s)
    f.close()
'''
