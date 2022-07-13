import numpy as np

def segmentation_ious(pred, label, classes):
    '''computes iou for one ground truth mask and predicted mask'''
    ious = []
    for c in classes:
        label_c = (label == c)
        pred_c = (pred == c)
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

if __name__ == "__main__":
    classes = [1, 2]
    label = np.zeros([10, 10])
    label[2:5, 2:5] = 1
    pred = np.zeros_like(label)
    pred[3:6, 3:6] = 1
    mean_iou = segmentation_ious(pred, label, classes)
    print(mean_iou)
