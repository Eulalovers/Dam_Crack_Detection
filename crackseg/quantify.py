"""裂缝几何量化：长度、宽度、走向、等级。

流程（每条裂缝 = 掩膜中的一个连通域）：
  1. 骨架化（skeletonize）得到 1 像素宽的中心线；
  2. 长度 = 骨架上相邻像素间距之和（水平/竖直相邻计 1，对角相邻计 √2）；
  3. 宽度：对掩膜做欧氏距离变换，骨架点处的距离值 d 即为该点到裂缝边缘的距离，
     宽度 ≈ 2d − 1（像素中心到边界的离散修正）；取最大值 / 均值 / 95 分位数；
  4. 走向：骨架点坐标做 PCA，主方向与水平方向的夹角；
  5. 像素 → 毫米：乘以 GSD（地面采样距离，mm/pixel）。

GSD（摄影测量中心投影关系，像面与坝面近似平行时）：
    GSD[mm/px] = 像元尺寸[μm] × 物距[m] / 焦距[mm]
例：DJI Zenmuse P1（4.4 μm，35 mm 镜头）距坝面 3 m → 4.4×3/35 ≈ 0.38 mm/px。
宽度测量的前提是裂缝在图像上至少约 2 个像素宽，因此要可靠测出 0.2 mm 的 A/B 类分界，
GSD 需要优于 ~0.1 mm/px —— 这直接决定了贴近摄影测量的航线（物距、镜头）设计。
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from .config import CRACK_GRADE_THRESHOLDS_MM


def gsd_mm(pixel_size_um: float, distance_m: float, focal_mm: float) -> float:
    return pixel_size_um * distance_m / focal_mm


def required_distance_m(target_gsd_mm: float, pixel_size_um: float, focal_mm: float) -> float:
    """反算：要达到目标 GSD，相机最远离坝面多少米。"""
    return target_gsd_mm * focal_mm / pixel_size_um


def grade_by_width(width_mm: float) -> str:
    for thr, g in CRACK_GRADE_THRESHOLDS_MM:
        if width_mm < thr:
            return g
    return CRACK_GRADE_THRESHOLDS_MM[-1][1]


@dataclass
class CrackInfo:
    id: int
    length_px: float
    max_width_px: float
    mean_width_px: float
    p95_width_px: float
    orientation_deg: float   # 0=水平, 90=竖直
    bbox_xywh: tuple
    centroid_xy: tuple
    length_mm: float = float("nan")
    max_width_mm: float = float("nan")
    mean_width_mm: float = float("nan")
    p95_width_mm: float = float("nan")
    grade: str = "-"

    def to_dict(self):
        return asdict(self)


def skeleton_orientation(skel: np.ndarray, radius: int = 5) -> np.ndarray:
    """骨架局部切线方向（弧度，图像坐标系，x 向右 y 向下）。
    对每个骨架点，统计半径 radius 邻域内骨架点坐标的二阶矩（结构张量），
    主方向 θ = 0.5·atan2(2Sxy, Sxx−Syy)。用卷积一次性算完所有点。"""
    r = np.arange(-radius, radius + 1, dtype=np.float32)
    dx, dy = np.meshgrid(r, r)
    disk = (dx ** 2 + dy ** 2 <= radius ** 2).astype(np.float32)
    s = skel.astype(np.float32)
    sxx = cv2.filter2D(s, -1, dx * dx * disk, borderType=cv2.BORDER_CONSTANT)
    syy = cv2.filter2D(s, -1, dy * dy * disk, borderType=cv2.BORDER_CONSTANT)
    sxy = cv2.filter2D(s, -1, dx * dy * disk, borderType=cv2.BORDER_CONSTANT)
    return 0.5 * np.arctan2(2 * sxy, sxx - syy)


def orthogonal_widths(mask: np.ndarray, ys, xs, theta, max_half=40.0, step=0.1):
    """正交剖面法：在每个骨架点沿法线方向双向亚像素采样（双线性插值），
    找到掩膜值降到 0.5 的位置作为左右边界，两侧距离之和即为垂直于走向的宽度。"""
    from scipy.ndimage import map_coordinates
    m = mask.astype(np.float32)
    t = np.arange(step, max_half, step, dtype=np.float32)
    nx, ny = -np.sin(theta), np.cos(theta)              # 法线方向
    widths = np.zeros(len(xs), np.float32)
    for sign in (1, -1):
        px = xs[:, None] + sign * nx[:, None] * t[None]
        py = ys[:, None] + sign * ny[:, None] * t[None]
        v = map_coordinates(m, [py.ravel(), px.ravel()], order=1, mode="constant").reshape(px.shape)
        below = v < 0.5
        idx = np.where(below.any(1), below.argmax(1), len(t) - 1)
        # 在跨越 0.5 的相邻两采样点之间线性插值，得到亚像素边界
        v1 = v[np.arange(len(xs)), idx]
        v0 = np.where(idx > 0, v[np.arange(len(xs)), np.maximum(idx - 1, 0)], 1.0)
        frac = np.clip((v0 - 0.5) / np.maximum(v0 - v1, 1e-6), 0, 1)
        widths += t[idx] - step + frac * step
    return widths


def junction_zone(skel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """骨架分叉点（8 邻域内骨架点 ≥ 3）及其周围 2 倍局部宽度范围。"""
    s = skel.astype(np.uint8)
    nb = cv2.filter2D(s, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s
    jy, jx = np.nonzero((s > 0) & (nb >= 3))
    zone = np.zeros(skel.shape, np.uint8)
    if len(jy):
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        for y, x in zip(jy, jx):
            cv2.circle(zone, (int(x), int(y)), int(np.ceil(2 * dist[y, x] + 2)), 1, -1)
    return zone > 0


def postprocess(mask: np.ndarray, min_area_px: int = 64) -> np.ndarray:
    """去除孤立小噪点（混凝土气孔、斑点等），填补 1 像素的小断口。"""
    m = mask.astype(bool)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return remove_small_objects(m, min_area_px).astype(np.uint8)


def quantify_cracks(mask: np.ndarray, gsd: float | None = None, min_length_px: float = 20.0):
    mask = (mask > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    skel_all = skeletonize(mask > 0)
    theta = skeleton_orientation(skel_all)
    sy, sx = np.nonzero(skel_all)
    th = theta[sy, sx]
    # 每个骨架点代表的弧长：走向为 θ 的数字直线上，一个像素对应 1/max(|cosθ|,|sinθ|) 的长度
    seg_len = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    # 搜索半径按最宽处自适应（距离变换最大值的 2 倍），避免宽裂缝 / 分叉处测宽被截断
    dmax = float(cv2.distanceTransform(mask, cv2.DIST_L2, 3).max()) if len(sy) else 0.0
    w_all = orthogonal_widths(mask, sy.astype(np.float32), sx.astype(np.float32), th,
                              max_half=max(10.0, 2.0 * dmax + 3.0))
    w_all = np.maximum(w_all, 1.0)
    # 分叉点附近法线会同时穿过多条分支，宽度会被高估：这些骨架点只计长度，不参与宽度统计
    near_junction = junction_zone(skel_all, mask)[sy, sx]
    comp_of = lab[sy, sx]
    order = np.argsort(comp_of, kind="stable")
    bounds = np.searchsorted(comp_of[order], np.arange(n + 1))
    cracks = []
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        sel = order[bounds[k]:bounds[k + 1]]
        length = float(seg_len[sel].sum())
        if length < min_length_px:
            continue
        keep = ~near_junction[sel]
        widths = w_all[sel][keep] if keep.sum() >= max(5, 0.2 * len(sel)) else w_all[sel]
        ys, xs = sy[sel], sx[sel]
        if len(xs) >= 2:
            cov = np.cov(np.stack([xs, -ys]).astype(np.float64))
            v = np.linalg.eigh(cov)[1][:, -1]
            ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180)
            ang = ang if ang <= 90 else 180 - ang
        else:
            ang = 0.0
        c = CrackInfo(len(cracks) + 1, length, float(widths.max()), float(widths.mean()),
                      float(np.percentile(widths, 95)), round(ang, 1), (int(x), int(y), int(w), int(h)),
                      (float(cents[k][0]), float(cents[k][1])))
        if gsd:
            c.length_mm = c.length_px * gsd
            # 最大宽度对单个骨架点的噪声敏感，分级用 95 分位宽度，更稳健
            c.max_width_mm, c.mean_width_mm, c.p95_width_mm = (c.max_width_px * gsd, c.mean_width_px * gsd,
                                                               c.p95_width_px * gsd)
            c.grade = grade_by_width(c.p95_width_mm)
        cracks.append(c)
    return cracks
