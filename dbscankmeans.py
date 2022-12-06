import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
plt.rc('font', family='Times New Roman')
'''
x1,y1 = datasets.make_circles(n_samples=500,factor=0.6,noise=0.05)
x2,y2 = datasets.make_blobs(n_samples=500,n_features=2,centers=[[1.2,1.2]], cluster_std=[[.1]])

x=np.concatenate((x1,x2))
plt.scatter(x[:,0],x[:,1])


y_pred = KMeans(n_clusters=3,random_state=9).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)  
plt.show()

y_pred = DBSCAN(eps = 0.16, min_samples = 10).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()
'''

x = np.linspace(0, 10, 200)
y1 = -0.01 * x ** 3 + 0.1 * x * x + 1 + np.random.random(x.shape)
y2 = -0.01 * x ** 3 + 0.1 * x * x + 6 + np.random.random(x.shape)
# plt.scatter([x, x], [y1, y2])
# plt.show()

x = np.concatenate((x, x))
y = np.concatenate((y1, y2))
xy = np.concatenate((x.reshape(400, 1), y.reshape(400, 1)),axis=1)

plt.subplot(1, 2, 1)
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(xy)
plt.scatter(xy[:,0], xy[:,1], c=y_pred, s=2)  
plt.title('Kmeans')

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.subplot(1, 2, 2)
y_pred = DBSCAN(eps = 1.0, min_samples = 5).fit_predict(xy)
plt.scatter(xy[:,0], xy[:,1], c=y_pred, s=2)
plt.title('DBSCAN')

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()
