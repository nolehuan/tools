import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

from scipy.interpolate import UnivariateSpline

points = []
with open('./TriangulatedPoint3d_lane0.txt', 'r') as f:
# with open('./lane1_part.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        point = line.split(' ')
        if len(point) == 3:
            points.append(point)
f.close()
points = np.array(points).astype('float64')
points = points[points[:, 2].argsort()]
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

fig = plt.figure('point cloud')
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z, 'r.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(1, 2)
plt.show()

points = []
with open('./processed_lane0_inliers.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        point = line.split(' ')
        if len(point) == 3:
            points.append(point)
f.close()
points = np.array(points).astype('float64')
points = points[points[:, 2].argsort()]
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# https://stackoverflow.com/questions/35273741/how-to-fit-a-line-through-a-3d-pointcloud
# new axis
u = np.arange(len(x))

# UnivariateSpline
s = 0.7 * len(u)     # smoothing factor
spx = UnivariateSpline(u, x, s=s)
spy = UnivariateSpline(u, y, s=s)
spz = UnivariateSpline(u, z, s=s)
#
xnew = spx(u)
ynew = spy(u)
znew = spz(u)

Z = z.reshape(-1, 1)
Z = np.hstack([np.ones_like(Z), Z, Z ** 2])
zzz = np.linspace(min(z), max(z), 326)
zzz.sort(axis=-1,kind='quicksort')
ZZZ = zzz.reshape(-1, 1)
ZZZ = np.hstack([np.ones_like(ZZZ), ZZZ, ZZZ ** 2])

ransac_x = linear_model.RANSACRegressor()
ransac_x.fit(Z, x)
line_x_ransac = ransac_x.predict(ZZZ)

ransac_y = linear_model.RANSACRegressor()
ransac_y.fit(Z, y)
line_y_ransac = ransac_y.predict(ZZZ)



fig = plt.figure('point cloud')
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(xnew, ynew, znew, 'r.')
ax.plot3D(x, y, z, 'b.')
ax.plot3D(line_x_ransac, line_y_ransac, zzz, 'g.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-5.5, -4.5)
plt.show()

t = np.random.uniform(0, 20, size = 200)
t.sort(axis=-1,kind='quicksort')
v = 1.5
x = v * t
y = 0.1 * np.random.random(size = 200) + 2
z = 0.001 * t + 0.01 * t * t + 0.1 * np.random.random(size = 200) + 2

# xyz = np.hstack(x, y, z)

# 绘制3D图
# fig = plt.figure()

# ax = plt.gca(projection='3d')
# ax.plot3D(x, y, z, 'b.')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_ylim3d(0, 5)
# ax.set_zlim3d(0, 5)

# plt.show()

# ## 最小二乘法求解方程
# # 创建系数矩阵 A 和矩阵 b
# A = np.ones((len(t), 3))
# b = np.zeros((len(t), 1))
# for j in range(len(t)):
#     A[j, 0] = x[j]
#     A[j, 1] = y[j]
#     b[j, 0] = z[j]

# # 通过 X=(AT*A)^(-1)*AT*b 直接求解
# A_T = A.T
# A1 = np.dot(A_T, A)
# A2 = np.linalg.inv(A1)
# A3 = np.dot(A2, A_T)
# X = np.dot(A3, b)
# print('平面拟合结果为: z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))

X = x.reshape(-1, 1)
# Y = y.reshape(-1, 1)
X = np.hstack([np.ones_like(X), X, X ** 2])
# XY = np.hstack([X, X ** 2, Y, Y ** 2])

xxx = np.linspace(0, 30, 200)
xxx.sort(axis=-1,kind='quicksort')
XXX = xxx.reshape(-1, 1)
XXX = np.hstack([np.ones_like(XXX), XXX, XXX ** 2])

# ransac = linear_model.RANSACRegressor()
# ransac.fit(XY, z)
# line_z_ransac = ransac.predict()

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
line_y_ransac = ransac.predict(XXX)

ransac = linear_model.RANSACRegressor()
ransac.fit(X, z)
line_z_ransac = ransac.predict(XXX)


fig = plt.figure('graph')
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(xxx, line_y_ransac, line_z_ransac, 'r.')
ax.plot3D(x, y, z, 'b.')
ax.plot(xxx, line_y_ransac, line_z_ransac, label='curve', c='g')

ax.set_xlim(0, 30)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)
plt.show()

