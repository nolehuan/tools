import numpy as np

def softNMS(bboxs, scores, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    n = bboxs.shape[0]
    x1 = bboxs[:,0]
    y1 = bboxs[:,1]
    x2 = bboxs[:,2]
    y2 = bboxs[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.array([np.arange(n)]).reshape(n,)

    for i in range(n):
        tbbox = bboxs[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        if i != n-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
        
        if tscore < maxscore:
            bboxs[i, :] = bboxs[maxpos + i + 1, :]
            bboxs[maxpos + i + 1, :] = tbbox
            tbbox = bboxs[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]
        
        x_tl = np.maximum(bboxs[i, 0], bboxs[pos:, 0])
        y_tl = np.maximum(bboxs[i, 1], bboxs[pos:, 1])
        x_br = np.minimum(bboxs[i, 2], bboxs[pos:, 2])
        y_br = np.minimum(bboxs[i, 3], bboxs[pos:, 3])
        
        w = np.maximum(0, x_br - x_tl + 1)
        h = np.maximum(0, y_br - y_tl + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)

        # 0-linear 1-gaussian 2-original nms
        if method == 0:
            weight = np.ones(iou.shape)
            weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
        elif method == 1:
            weight = np.exp(-(iou * iou) / sigma)
        else:
            weight = np.ones(iou.shape)
            weight[iou > Nt] = 0
        
        scores[pos:] = weight * scores[pos:]
    
    indices = indices[scores > thresh]
    keep = indices.astype(int)

    return keep

if __name__ == '__main__':
    bboxs = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400]], dtype=np.float32)
    scores = np.array([0.69, 0.48, 0.87, 0.96], dtype=np.float32)
    keep = softNMS(bboxs, scores, thresh=0.1, method=1)
    print(keep)


