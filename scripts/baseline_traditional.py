"""传统图像处理基线：CLAHE 增强 → 黑帽变换突出暗色细线 → Otsu 自动阈值 → 形态学去噪。

这是工程上无需训练的常用做法，用来说明为什么要用深度学习：
混凝土表面的气孔、水渍、阴影、析出物同样是"暗色"，阈值法无法区分，误检严重。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.config import RUNS_DIR  # noqa: E402
from crackseg.datasets import deepcrack_splits, load_pair, ozgenel_splits  # noqa: E402
from crackseg.metrics import SegEvaluator  # noqa: E402
from crackseg.quantify import postprocess  # noqa: E402


def traditional_segment(img_rgb, kernel=15, min_area=64):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    _, m = cv2.threshold(bh, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return postprocess(m, min_area)


def main():
    for ds, (_, _, test), kernel, area in [("deepcrack", deepcrack_splits(), 15, 64),
                                           ("ozgenel", ozgenel_splits(), 31, 400)]:
        ev = SegEvaluator()
        for pair in test:
            img, gt = load_pair(pair)
            ev.update(traditional_segment(img, kernel, area).astype(np.float32), gt)
        res = ev.result()
        res.update(dataset=ds, model="traditional", loss="otsu", seed=0, train_minutes=0)
        out = RUNS_DIR / ds / "traditional_otsu_s0"
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
        print(ds, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}))


if __name__ == "__main__":
    main()
