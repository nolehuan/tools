import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

x = np.random.uniform(-5, 5, size = 100)
print(x)
X = x.reshape(-1, 1)
print(X)
X = np.hstack([X, X ** 2])
print(X)
y = 0.25 * x ** 2 + x - 5 + np.random.normal(0, 1, 100)
print(y)
# plt.scatter(x, y)
# plt.show()

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
line_y_ransac = ransac.predict(X)
print(line_y_ransac)
plt.scatter(x, y)
plt.plot(np.sort(x), line_y_ransac[np.argsort(x)], color='red')
# plt.plot(x, line_y_ransac, color='red')
plt.show()
