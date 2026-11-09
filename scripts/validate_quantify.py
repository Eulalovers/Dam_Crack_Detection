"""用已知真值的合成裂缝验证量化算法（长度、宽度）的精度。

生成水平、45° 斜向、蜿蜒（正弦）三种形态，宽度 3~15 像素的裂缝，
比较算法测得值与真值的相对误差。结果写入 docs/quantify_validation.md。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.quantify import quantify_cracks  # noqa: E402


def make_crack(shape, width, size=600):
    t = np.linspace(0, 1, 4000)
    if shape == "horizontal":
        x, y = 50 + 500 * t, np.full_like(t, 300.0)
    elif shape == "diagonal":
        x, y = 80 + 440 * t, 80 + 440 * t
    else:  # sine：蜿蜒裂缝
        x, y = 50 + 500 * t, 300 + 60 * np.sin(2 * np.pi * 2.5 * t)
    true_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    # 以中心线距离生成掩膜：到中心线距离 ≤ width/2 的像素属于裂缝
    center = np.ones((size, size), bool)
    center[np.round(y).astype(int), np.round(x).astype(int)] = False
    # 用亚像素采样的中心线点计算距离，避免中心线栅格化误差
    dist = ndimage.distance_transform_edt(center)
    mask = (dist <= (width - 1) / 2).astype(np.uint8)
    return mask, true_len


def main():
    rows = []
    for shape in ["horizontal", "diagonal", "sine"]:
        for w in [3, 5, 7, 9, 11, 15]:
            mask, true_len = make_crack(shape, w)
            c = max(quantify_cracks(mask), key=lambda c: c.length_px)
            # 栅格化后的真实平均宽度 = 掩膜面积 / 中心线真实长度（斜向时小于标称宽度）
            true_w = mask.sum() / true_len
            rows.append((shape, true_w, true_len, c.length_px, c.mean_width_px, c.p95_width_px))
    lines = ["| 形态 | 真实平均宽度(px) | 测得平均宽度 | 宽度误差 | 真实长度 | 测得长度 | 长度误差 |",
             "|---|---|---|---|---|---|---|"]
    werr, lerr = [], []
    for s, w, tl, l, mw, pw in rows:
        we, le = (mw - w) / w * 100, (l - tl) / tl * 100
        werr.append(abs(we))
        lerr.append(abs(le))
        lines.append(f"| {s} | {w:.2f} | {mw:.2f} | {we:+.1f}% | {tl:.0f} | {l:.0f} | {le:+.1f}% |")
    lines.append(f"\n平均绝对误差：宽度 {np.mean(werr):.1f}%，长度 {np.mean(lerr):.1f}%")
    out = "\n".join(lines)
    print(out)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/quantify_validation.md").write_text("# 量化算法合成数据验证\n\n" + out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
