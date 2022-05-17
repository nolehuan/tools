# y gt, y_hat pred
# accuracy
# sum(y == y_hat) / y.size
# precision
# sum((y_hat == 1) & (y == 1)) / sum(y_hat == 1)
# recall
# sum((y_hat == 1) & (y == 1)) / sum(y == 1)
# f1
# 2pr/(p + r)
# auc & roc

# AP

# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/utils.py
# import tqdm
import numpy as np

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:         true positive (np array), true positive 1, false positive 0
        conf:       Objectness value from 0-1 (np array).
        pred_cls:   Predicted object classes (np array).
        target_cls: True object classes (np array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (np.array).
        precision: The precision curve (np.array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[:-1] != mrec[1:])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    tp = np.array([0, 1, 1, 1, 0, 1, 0])
    conf = np.array([0.45, 0.6, 0.86, 0.94, 0.23, 0.76, 0.34])
    pred_cls = np.array([0, 1, 2, 1, 2, 3, 0])
    target_cls = np.array([3, 1, 2, 1, 0, 3, 1])
    p, r, ap, f1, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls)
    print(p)
    print(r)
    print(ap)
    # print(f1)
    # print(unique_classes)

