import numpy as np
from math import sqrt

def distElud(vecA, vecB):
    return sqrt(sum(np.power((vecA - vecB), 2)))

def randCent(data, k):
    n = np.shape(data)[1]
    center = np.mat(np.zeros((k, n)))
    for j in range(n):
        rangeJ = float(max(data[:, j]) - min(data[:, j]))
        center[:, j] = min(data[:, j]) + rangeJ * np.random.rand(k, 1)
    return center

def kmeans(data, k, dist = distElud, createCent = randCent):
    m = np.shape(data)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    center = createCent(data, k)
    # print(center)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = dist(data[i, :], np.array(center[j, :]).squeeze(0))
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2
        for cent in range(k):
            # 将矩阵转化为 array 数组类型
            dataCent = data[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # print(dataCent)
            if dataCent.shape[0] == 0:
                continue
            center[cent, :] = np.mean(dataCent, axis = 0)
    return center, clusterAssment

if __name__ == "__main__":
    data = np.array([[1, 2], [2, 1], [3, 1], [5, 4], [5, 5], [6, 5],
                    [10, 8], [7, 9], [11, 5], [14, 9], [14, 14]])
    center, clusterAssment = kmeans(data, 3)
    print(center)
    print(clusterAssment)
