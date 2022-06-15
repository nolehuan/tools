# feature_visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font', family='Times New Roman')

df = pd.read_csv("./glcm_features.csv")
array = df.values
x_feature = array[:,2:]
y_label = array[:,1].astype('int')


con = x_feature[:, 0]
hom = x_feature[:, 2]
cor = x_feature[:, 4]
asm = x_feature[:, 6]
dis = x_feature[:, 8]

fig = plt.figure('graph')
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(con[:200], asm[:200], cor[:200], 'r.', label = 'wet')
ax.plot3D(con[200:400], asm[200:400], cor[200:400], 'g.', label = 'dry')
ax.plot3D(con[400:], asm[400:], cor[400:], 'b.', label = 'icy')

# ax.set_facecolor("red")

ax.set_xlabel('Contrast')
ax.set_ylabel('ASM')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Correlation', rotation = 90)
# 坐标区域背景透明
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


plt.legend(loc = 'best')
plt.show()

fig.savefig('feature.pdf', dpi=1200, format='pdf')
