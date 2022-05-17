import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 设置图例字号
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()

# 设置三维图形模式
ax = fig.gca(projection='3d')

# 测试数据
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-4, 4, 100) / 4
r = z**3 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# 绘制图形
ax.plot(x, y, z, label='parametric curve')

# 显示图例
ax.legend()

# 显示图形
plt.show()
