"""裂缝分割数据集。

DeepCrack: 训练集 300 张 / 测试集 237 张，544x384，混凝土表面裂缝，像素级标注。
从训练集中按固定随机种子划出 30 张作为验证集，用于选模型；测试集只用于最终报告。
"""
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEEPCRACK_DIR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pairs(img_dir: Path, lab_dir: Path):
    pairs = []
    for img in sorted(img_dir.glob("*.jpg")):
        lab = lab_dir / (img.stem + ".png")
        if lab.exists():
            pairs.append((img, lab))
    return pairs


def deepcrack_splits(val_count=30, seed=0):
    train = _pairs(DEEPCRACK_DIR / "train_img", DEEPCRACK_DIR / "train_lab")
    test = _pairs(DEEPCRACK_DIR / "test_img", DEEPCRACK_DIR / "test_lab")
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    val = [train[i] for i in idx[:val_count]]
    tr = [train[i] for i in idx[val_count:]]
    return tr, val, test


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def normalize(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def train_augment(crop=384):
    # 增强策略针对大坝巡检场景：光照不均、阴影、潮湿反光、成像噪声、拍摄角度任意
    return A.Compose([
        A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3),
            A.RandomGamma((70, 150)),
            A.CLAHE(clip_limit=3.0),
        ], p=0.8),
        A.RandomShadow(p=0.2),                       # 模拟坝面阴影
        A.HueSaturationValue(10, 20, 15, p=0.3),     # 水渍、青苔导致的色偏
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.06)),
            A.GaussianBlur(blur_limit=(3, 5)),       # 轻微失焦/运动模糊
            A.ImageCompression(quality_range=(50, 95)),
        ], p=0.4),
    ])


# ---------------------------------------------------------------------------
# 高分辨率混凝土墙面数据集（Özgenel, Mendeley jwsn7tfbrp）：458 张 4032x3024 等原图。
# 先运行 scripts/prepare_ozgenel.py 把 JPEG 解码成 .npy 缓存（随机裁剪时读取更快），
# 并生成固定的 train/val/test 划分。
# ---------------------------------------------------------------------------
OZ_CACHE = DEEPCRACK_DIR.parent.parent / "cache" / "ozgenel"


def ozgenel_splits():
    import json
    split = json.loads((OZ_CACHE / "split.json").read_text())
    to_pairs = lambda names: [(OZ_CACHE / f"{n}_img.npy", OZ_CACHE / f"{n}_mask.npy") for n in names]
    return to_pairs(split["train"]), to_pairs(split["val"]), to_pairs(split["test"])


class LargeImageCropDataset(Dataset):
    """从高分辨率原图中随机裁剪训练块。70% 的裁剪以裂缝像素为中心（正样本偏置采样），
    缓解"裂缝只占 0.5%~1% 像素"导致的大量空白块问题；30% 完全随机，学习背景干扰。"""

    def __init__(self, pairs, crop=512, samples_per_epoch=2400, pos_ratio=0.7):
        self.pairs, self.crop, self.n, self.pos_ratio = pairs, crop, samples_per_epoch, pos_ratio
        self.aug = train_augment(crop)
        self._pos = {}

    def __len__(self):
        return self.n

    def _crack_coords(self, i, mask):
        if i not in self._pos:
            ys, xs = np.nonzero(mask[::4, ::4])
            self._pos[i] = np.stack([ys * 4, xs * 4], 1) if len(ys) else None
        return self._pos[i]

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        i = int(rng.integers(len(self.pairs)))
        img = np.load(self.pairs[i][0], mmap_mode="r")
        mask = np.load(self.pairs[i][1], mmap_mode="r")
        h, w = mask.shape
        c = self.crop
        coords = self._crack_coords(i, mask) if rng.random() < self.pos_ratio else None
        if coords is not None:
            cy, cx = coords[rng.integers(len(coords))] + rng.integers(-c // 3, c // 3 + 1, 2)
            y0, x0 = int(np.clip(cy - c // 2, 0, h - c)), int(np.clip(cx - c // 2, 0, w - c))
        else:
            y0, x0 = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
        im = np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c])
        m = np.ascontiguousarray(mask[y0:y0 + c, x0:x0 + c])
        out = self.aug(image=im, mask=m)
        return normalize(out["image"]), torch.from_numpy(out["mask"][None].astype(np.float32)), str(i)


# ---------------------------------------------------------------------------
# 真实混凝土大坝数据集（Gharehbaghi et al., Data in Brief 2026，CC BY 4.0）
#   DamCrack  : 无人机 + 手机拍摄，4000 张 640x640，Zenodo 17274707
#   DamSegment: 1500 张 640x640，按难度分 Easy / Medium / Hard，Mendeley z5z6gtt5t4
# 掩膜为 RGB：红色 = 裂缝，蓝色 = 剥落。本项目做裂缝分割，剥落视为背景。
# 相邻编号的切片可能来自同一张原图，为避免训练/测试泄漏，按编号连续分段划分（70/10/20），
# 测试集相当于"未见过的坝段"。
# ---------------------------------------------------------------------------
DAM_DIR = DEEPCRACK_DIR.parent


def _num(p: Path):
    import re
    return int(re.findall(r"\d+", p.stem)[0])


def _block_split(items, ratios=(0.7, 0.1, 0.2)):
    n = len(items)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return items[:a], items[a:b], items[b:]


def dam_splits():
    """返回 train, val, test；test 为 {子集名: pairs} 字典，便于分别报告。"""
    dc_root = DAM_DIR / "damcrack" / "Damage Segmentaion"
    dc = sorted((dc_root / "Images").glob("*.jpg"), key=_num)
    dc = [(p, dc_root / "Refined Labels" / "Mask" / (p.stem + ".png")) for p in dc]
    tr, va, te = _block_split(dc)
    test = {"DamCrack": te}
    ds_root = DAM_DIR / "damsegment" / "Damage Segmentaion"
    for lv in ["Easy", "Medium", "Hard"]:
        ps = sorted((ds_root / lv / "Images").glob("*.jpg"), key=_num)
        ps = [(p, ds_root / lv / "Labels" / "Mask" / (p.stem + "_mask.png")) for p in ps]
        a, b, c = _block_split(ps)
        tr, va = tr + a, va + b
        test[f"DamSegment-{lv}"] = c
    return tr, va, test


def read_dam_mask(path):
    m = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return ((m[..., 0] > 127) & (m[..., 1] < 128) & (m[..., 2] < 128)).astype(np.uint8)


def load_pair(pair):
    """读取 (图像, 掩膜)，兼容 jpg/png 原始文件、大坝 RGB 掩膜和 npy 缓存。"""
    img_p, lab_p = pair
    if str(img_p).endswith(".npy"):
        return np.load(img_p), np.load(lab_p)
    if "damcrack" in str(lab_p) or "damsegment" in str(lab_p):
        return read_image(img_p), read_dam_mask(lab_p)
    return read_image(img_p), read_mask(lab_p)


class CrackDataset(Dataset):
    def __init__(self, pairs, train=False, crop=384, repeat=1):
        self.pairs = pairs
        self.train = train
        self.aug = train_augment(crop) if train else None
        self.repeat = repeat if train else 1

    def __len__(self):
        return len(self.pairs) * self.repeat

    def __getitem__(self, i):
        img_p, lab_p = self.pairs[i % len(self.pairs)]
        img, mask = load_pair((img_p, lab_p))
        if self.aug is not None:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return normalize(img), torch.from_numpy(mask[None].astype(np.float32)), str(img_p.name)
