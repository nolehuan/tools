import numpy as np

def createData():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(input, data, label, k):
    dataSize = data.shape[0]
    assert(k <= dataSize)
    diff = np.tile(input, (dataSize, 1)) - data
    sqdiff = diff ** 2
    squareDist = np.sum(sqdiff, axis = 1)
    dist = squareDist ** 0.5
    sortedDistIndex = np.argsort(dist)

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

if __name__ == "__main__":
    input = np.array([1.1, 0.9])
    data, label = createData()
    classes = classify(input, data, label, 2)
    print(classes)

