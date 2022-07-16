import torch
import math

def box_ciou(b1, b2):
    """
    b1: tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), x y w h
    b2: tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), x y w h
    ciou: tensor, shape = (batch, feat_w, feat_h, anchor_num, 1)
    """

    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    intersect_min = torch.max(b1_min, b2_min)
    intersect_max = torch.min(b1_max, b2_max)
    intersect_wh = torch.max(intersect_max - intersect_min, torch.zeros_like(intersect_max))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    enclose_min = torch.min(b1_min, b2_min)
    enclose_max = torch.max(b1_max, b2_max)
    enclose_wh = torch.max(enclose_max - enclose_min, torch.zeros_like(enclose_max))

    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) \
        - torch.atan(b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return ciou

box1 = torch.tensor([[2., 2., 1., 1.]])
box2 = torch.tensor([[2., 2., 2., 2.]])
print(box_ciou(box1, box2))
