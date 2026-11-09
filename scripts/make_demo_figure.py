"""拼接演示图：原图 | 人工标注 | 模型预测（读取 infer.py 输出的 *_mask.png）。
数据集没有拍摄距离/相机参数，无法得到真实 GSD，因此演示图只比较分割结果，不显示毫米宽度和等级。

用法：python scripts/make_demo_figure.py --pred_dir G:/Dam_crack_data/outputs/demo_dam --out docs/images/dam_demo.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.datasets import dam_splits, load_pair  # noqa: E402


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def overlay(img, m, color):
    out = img.copy()
    out[m] = (0.35 * out[m] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out", default="docs/images/dam_demo.jpg")
    ap.add_argument("--size", type=int, default=420)
    a = ap.parse_args()
    pred_dir = Path(a.pred_dir)
    _, _, test = dam_splits()
    lookup = {}
    for subset, items in test.items():
        for p in items:
            img_path = p[0] if isinstance(p, tuple) else p
            lookup[Path(img_path).stem] = (subset, p)

    rows = []
    for mk in sorted(pred_dir.glob("*_mask.png")):
        stem = mk.name[:-len("_mask.png")]
        if stem not in lookup:
            continue
        subset, p = lookup[stem]
        img, gt = load_pair(p)
        pred = cv2.imread(str(mk), cv2.IMREAD_GRAYSCALE) > 127
        tiles = [label(img, f"{subset}  {stem}"), label(overlay(img, gt > 0, (255, 0, 0)), "Ground truth"),
                 label(overlay(img, pred, (0, 200, 255)), "Prediction (unet_dsc)")]
        rows.append(np.hstack([cv2.resize(t, (a.size, a.size), interpolation=cv2.INTER_AREA) for t in tiles]))
    fig = np.vstack(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(fig, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"saved {a.out}  {fig.shape}")


if __name__ == "__main__":
    main()
