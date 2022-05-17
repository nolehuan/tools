# https://zhuanlan.zhihu.com/p/72347472
import numpy as np
import torch
import torch.nn as nn

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sm = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return sm

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1.-epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce

def to_onehot(label, n_classes):
    N = label.shape[0]
    onehot = np.zeros((N, n_classes))
    for i in range(N):
        onehot[i][label[i]] = 1
    return onehot


if __name__ == '__main__':
    x = np.array([[5, 2.5, 0.1], [4, 0.8, -2], [-2, 1.5, -30]])
    label = np.array([0, 0, 1])
    n_classes = x.shape[1]
    targets = to_onehot(label, n_classes)
    predictions = softmax(x)
    print(predictions)
    ce = cross_entropy(predictions, targets)
    print(ce)

    loss = nn.CrossEntropyLoss()
    ce_torch = loss(torch.tensor(x), torch.tensor(label).type(torch.LongTensor))
    print(ce_torch)

