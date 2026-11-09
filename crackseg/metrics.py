"""评价指标。

像素级：
  - P / R / F1 / IoU @0.5：固定阈值 0.5，在整个测试集上累计 TP/FP/FN 后计算
  - ODS：整个数据集使用同一个最优阈值时的 F1（裂缝检测领域通用指标）
  - OIS：每张图各自取最优阈值时的平均 F1
拓扑 / 连续性（直接衡量"断断续续"问题）：
  - clDice：预测骨架落在真值内的比例 与 真值骨架被预测覆盖的比例 的调和平均
  - 断裂数 Frag：每条真值裂缝（连通域）被预测成了几段，理想值 1.0，越大越碎
  - β0 误差：|预测连通域数 − 真值连通域数|，同时反映断裂和孤立噪点误检
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def _f1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-9))


def cl_dice(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    sp, sg = skeletonize(pred > 0), skeletonize(gt > 0)
    tprec = (sp & (gt > 0)).sum() / max(sp.sum(), 1)
    tsens = (sg & (pred > 0)).sum() / max(sg.sum(), 1)
    return 2 * tprec * tsens / max(tprec + tsens, 1e-9)


def fragmentation(pred, gt, min_gt_len=20, tol=2):
    """每条真值裂缝被预测分成的段数（只统计被检测到的真值裂缝）。"""
    n_gt, lab_gt = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    _, lab_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    counts = []
    for k in range(1, n_gt):
        comp = (lab_gt == k).astype(np.uint8)
        if skeletonize(comp > 0).sum() < min_gt_len:
            continue
        region = cv2.dilate(comp, kernel) > 0
        ids = np.unique(lab_p[region])
        ids = ids[ids > 0]
        if len(ids):
            counts.append(len(ids))
    return counts


def betti0_error(pred, gt):
    n_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)[0] - 1
    n_g = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)[0] - 1
    return abs(n_p - n_g)


class SegEvaluator:
    def __init__(self, topo=True):
        self.topo = topo
        self.tp = np.zeros(len(THRESHOLDS))
        self.fp = np.zeros(len(THRESHOLDS))
        self.fn = np.zeros(len(THRESHOLDS))
        self.ois, self.cldice, self.frag, self.betti = [], [], [], []

    def update(self, prob, gt):
        gt = gt > 0
        best = 0.0
        for i, t in enumerate(THRESHOLDS):
            pb = prob >= t
            tp = np.logical_and(pb, gt).sum()
            fp = pb.sum() - tp
            fn = gt.sum() - tp
            self.tp[i] += tp
            self.fp[i] += fp
            self.fn[i] += fn
            best = max(best, _f1(tp, fp, fn)[2])
        self.ois.append(best)
        if self.topo:
            pred = (prob >= 0.5).astype(np.uint8)
            self.cldice.append(cl_dice(pred, gt))
            self.frag.extend(fragmentation(pred, gt))
            self.betti.append(betti0_error(pred, gt))

    def result(self):
        i5 = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
        p, r, f = _f1(self.tp[i5], self.fp[i5], self.fn[i5])
        iou = self.tp[i5] / max(self.tp[i5] + self.fp[i5] + self.fn[i5], 1)
        f1s = [_f1(a, b, c)[2] for a, b, c in zip(self.tp, self.fp, self.fn)]
        k = int(np.argmax(f1s))
        out = dict(P=p, R=r, F1=f, IoU=iou, ODS=f1s[k], ODS_thr=float(THRESHOLDS[k]),
                   OIS=float(np.mean(self.ois)))
        if self.topo:
            out.update(clDice=float(np.mean(self.cldice)),
                       Frag=float(np.mean(self.frag)) if self.frag else float("nan"),
                       Betti0=float(np.mean(self.betti)))
        return {k_: float(v) for k_, v in out.items()}
