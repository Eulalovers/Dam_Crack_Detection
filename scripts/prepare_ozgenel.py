"""把 Özgenel 高分辨率数据集解码为 .npy 缓存并生成固定划分（70% / 10% / 20%）。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.config import OZGENEL_DIR  # noqa: E402
from crackseg.datasets import OZ_CACHE  # noqa: E402


def convert(p: Path):
    name = p.stem
    out_i, out_m = OZ_CACHE / f"{name}_img.npy", OZ_CACHE / f"{name}_mask.npy"
    if not (out_i.exists() and out_m.exists()):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(OZGENEL_DIR / "BW" / p.name), cv2.IMREAD_GRAYSCALE)
        assert m.shape == img.shape[:2], name
        np.save(out_i, img)
        np.save(out_m, (m > 127).astype(np.uint8))  # 掩膜为 JPEG 格式，有压缩噪声，取 127 阈值
    return name


def main():
    OZ_CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted((OZGENEL_DIR / "rgb").glob("*.jpg"))
    with ThreadPoolExecutor(8) as ex:
        names = list(ex.map(convert, files))
    rng = random.Random(0)
    rng.shuffle(names)
    n = len(names)
    n_tr, n_val = int(n * 0.7), int(n * 0.1)
    split = dict(train=sorted(names[:n_tr]), val=sorted(names[n_tr:n_tr + n_val]), test=sorted(names[n_tr + n_val:]))
    (OZ_CACHE / "split.json").write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
