"""损失函数。

- BCE：基线。裂缝像素通常只占 1%~5%，BCE 会被大量背景主导，细裂缝容易被"学没"。
- Dice：直接优化前景重叠度，对类别不平衡不敏感。
- Focal：降低易分背景像素的权重，聚焦难分像素（裂缝边缘、低对比度细缝）。
- clDice（Shit et al., CVPR 2021）：在软骨架（中心线）上计算 Dice，惩罚"断裂"
  比惩罚"边缘略粗/略细"更重，直接约束拓扑连续性 —— 断一处就少一段中心线。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x, iters=10):
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def dice_loss(prob, target, eps=1.0):
    inter = (prob * target).sum((1, 2, 3))
    union = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def focal_loss(logits, target, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


def cldice_loss(prob, target, iters=10, eps=1.0):
    skel_p = soft_skeleton(prob, iters)
    skel_t = soft_skeleton(target, iters)
    tprec = ((skel_p * target).sum() + eps) / (skel_p.sum() + eps)
    tsens = ((skel_t * prob).sum() + eps) / (skel_t.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


class CrackLoss(nn.Module):
    def __init__(self, kind="combo", w_dice=1.0, w_focal=10.0, w_cl=0.5):
        super().__init__()
        self.kind, self.w_dice, self.w_focal, self.w_cl = kind, w_dice, w_focal, w_cl

    def forward(self, logits, target):
        if self.kind == "bce":
            return F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        loss = self.w_dice * dice_loss(prob, target) + self.w_focal * focal_loss(logits, target)
        if self.kind == "combo":
            loss = loss + self.w_cl * cldice_loss(prob, target)
        return loss  # kind == "dicefocal": 不含 clDice，用于拆分 clDice 的贡献
