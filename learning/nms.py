import numpy as np

def NMS(bboxs, score, thresh):
    x1 = np.array(bboxs[:, 0])
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    n = bboxs.shape[0]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep = []
    suppressed = np.array([0] * n)

    for i_ in range(n):
        i = order[i_]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for j_ in range(i_ + 1, n):
            j = order[j_]
            if suppressed[j] == 1:
                continue
            x_tl = max(x1[i], x1[j])
            y_tl = max(y1[i], y1[j])
            x_br = min(x2[i], x2[j])
            y_br = min(y2[i], y2[j])
            w = max(0, x_br - x_tl + 1)
            h = max(0, y_br - y_tl + 1)
            intersection = w * h
            iou = intersection / (area[i] + area[j] - intersection)
            if iou >= thresh:
                suppressed[j] = 1
    return keep


import matplotlib.pyplot as plt

if __name__ == '__main__':
    bboxs = np.array([[100, 100, 200, 200],
                    [100, 200, 200, 300],
                    [200, 200, 300, 300],
                    [50, 100, 200, 200],
                    [200, 200, 350, 400],
                    [200, 300, 300, 500],
                    [300, 200, 450, 450],
                    [200, 150, 500, 200],
                    [350, 250, 500, 450]])
    # bboxs = np.array([[100, 100, 200, 200],
    #                 [150, 150, 250, 250]])

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    for i in range(bboxs.shape[0]):
        rect = plt.Rectangle((bboxs[i][0], bboxs[i][1]), bboxs[i][2]-bboxs[i][0], bboxs[i][3]-bboxs[i][1], color='b', alpha=0.5)
        ax.add_patch(rect)
    plt.xlim(0,1000)
    plt.ylim(0,1000)
    ax.set_aspect('equal')
    plt.show()
    score = np.array([0.72, 0.87, 0.92, 0.73, 0.81, 0.93, 0.95, 0.85, 0.78])
    # score = np.array([0.72, 0.87])

    res = NMS(bboxs, score, 0.1)
    print(res)

