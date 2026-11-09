"""高分辨率影像的滑窗切片推理。

无人机/相机原图常为 4000x3000 甚至 8000x5000 像素，不能整图送入网络（显存不足），
直接缩小又会让 0.x mm 级细裂缝消失。做法：
  1. 按 tile 大小、overlap 重叠切块，保持原始分辨率；
  2. 每块的预测乘以中心高、边缘低的余弦权重窗，再累加回整图并按权重归一化，
     消除切块边缘处因上下文不足产生的接缝和断裂。
"""
import numpy as np
import torch

from .datasets import IMAGENET_MEAN, IMAGENET_STD


def _weight_window(tile):
    r = np.hanning(tile + 2)[1:-1]
    w = np.outer(r, r).astype(np.float32)
    return np.clip(w, 1e-3, None)


def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride)) + [length - tile]
    return sorted(set(s))


@torch.no_grad()
def predict_full(model, img_rgb, device="cuda"):
    """整图推理（图像较小时），自动 pad 到 32 的倍数。"""
    h, w = img_rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    img = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prob = torch.sigmoid(model(x).float())
    return prob[0, 0, :h, :w].float().cpu().numpy()


@torch.no_grad()
def predict_tiled(model, img_rgb, tile=512, overlap=128, batch=16, device="cuda"):
    h, w = img_rgb.shape[:2]
    if h < tile or w < tile:
        return predict_full(model, img_rgb, device)
    win = _weight_window(tile)
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    stride = tile - overlap
    coords = [(y, x) for y in _starts(h, tile, stride) for x in _starts(w, tile, stride)]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = np.stack([img_rgb[y:y + tile, x:x + tile] for y, x in chunk])
        x_t = torch.from_numpy(tiles).to(device).permute(0, 3, 1, 2).float() / 255.0
        x_t = (x_t - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prob = torch.sigmoid(model(x_t).float()).squeeze(1).cpu().numpy()
        for (y, x), p in zip(chunk, prob):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    return acc / np.maximum(wsum, 1e-6)
