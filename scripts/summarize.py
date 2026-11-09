"""汇总 runs/ 下所有实验的测试指标，输出 docs/results.md。

- 公开数据集（deepcrack / ozgenel）与大坝数据集（dam）分别出表；
- 同一配置的多个随机种子给出 均值±标准差；
- 大坝数据另外给出：分子集（DamCrack / DamSegment 难度）结果、迁移实验（零样本 / 少样本 / 预训练初始化）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.config import RUNS_DIR  # noqa: E402

NAMES = {
    ("traditional", "otsu"): "传统方法（CLAHE + 黑帽 + Otsu + 形态学）",
    ("unet", "bce"): "U-Net（BCE）基线",
    ("unet", "dicefocal"): "U-Net + Dice/Focal",
    ("unet", "combo"): "U-Net + Dice/Focal/clDice",
    ("unet_sp", "combo"): "+ 条状池化 SP",
    ("unet_dsc", "combo"): "+ 蛇形卷积 DSC",
    ("unet_sp_dsc", "combo"): "+ SP + DSC（完整方法）",
}
COLS = ["P", "R", "F1", "IoU", "ODS", "OIS", "clDice", "Frag", "Betti0"]
SUBSET_COLS = ["F1", "IoU", "clDice", "Frag"]


def init_src(r):
    init = r.get("init") or ""
    for s in ["ozgenel", "deepcrack"]:
        if f"/{s}/" in init.replace("\\", "/"):
            return s
    return "imagenet"


def fmt(vals):
    v = np.array(vals, dtype=float)
    return f"{np.nanmean(v):.4f}" + (f"±{np.nanstd(v):.4f}" if len(v) > 1 else "")


def ablation_table(groups, ds):
    out = ["| 方法 | 种子数 | " + " | ".join(COLS) + " | 训练(min) |", "|---|---|" + "---|" * (len(COLS) + 1)]
    for key, name in NAMES.items():
        runs = groups.get((ds, *key, 1.0, "imagenet"))
        if not runs:
            continue
        cells = [fmt([r[c] for r in runs]) for c in COLS]
        t = np.mean([r.get("train_minutes", 0) for r in runs])
        out.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + f" | {t:.0f} |")
    return out


def main():
    groups = defaultdict(list)
    for f in RUNS_DIR.glob("*/*/test_metrics.json"):
        if "smoke" in f.parent.name:
            continue
        r = json.loads(f.read_text())
        groups[(r["dataset"], r["model"], r["loss"], float(r.get("train_frac", 1.0)), init_src(r))].append(r)

    out = ["# 实验结果（测试集）\n",
           "指标：P/R/F1/IoU 为阈值 0.5 的像素级指标；ODS/OIS 为最优阈值 F1；clDice 为中心线重叠度；",
           "Frag = 每条真实裂缝被预测成几段（越接近 1 越连续）；Betti0 = 连通域个数误差（越小越好）。",
           "多随机种子时给出 均值±标准差。\n"]

    # ---------------- 大坝数据（主实验）----------------
    if any(k[0] == "dam" for k in groups):
        out.append("\n## 一、真实大坝数据（DamCrack + DamSegment）消融实验\n")
        out += ablation_table(groups, "dam")

        out.append("\n### 分子集 F1 / clDice / Frag\n")
        subsets = ["DamCrack", "DamSegment-Easy", "DamSegment-Medium", "DamSegment-Hard"]
        out.append("| 方法 | " + " | ".join(subsets) + " |")
        out.append("|---|" + "---|" * len(subsets))
        for key, name in NAMES.items():
            runs = groups.get(("dam", *key, 1.0, "imagenet"))
            if not runs or "subsets" not in runs[0]:
                continue
            cells = []
            for s in subsets:
                cells.append(" / ".join(fmt([r["subsets"][s][c] for r in runs]).split("±")[0]
                                        for c in ["F1", "clDice", "Frag"]))
            out.append(f"| {name} | " + " | ".join(cells) + " |")

        out.append("\n## 二、迁移实验：公开混凝土数据 → 大坝\n")
        out.append("| 设置 | 模型 | 大坝训练数据 | 初始化 | " + " | ".join(COLS[:7]) + " |")
        out.append("|---|---|---|---|" + "---|" * 7)
        rows = []
        for (ds, model, loss, frac, src), runs in groups.items():
            if ds != "dam":
                continue
            if loss.startswith("zeroshot"):
                setting, data, init = "零样本（不微调）", "0", loss.split("-", 1)[1]
            elif frac < 1 or src != "imagenet" or model == "unet_sp_dsc":
                setting = "少样本" if frac < 1 else "全量"
                data, init = f"{frac:.0%}", src
            else:
                continue
            r0 = runs if loss.startswith("zeroshot") else [r for r in runs if r["seed"] == 0] or runs
            rows.append((setting, model, data, init, [fmt([r[c] for r in r0]) for c in COLS[:7]]))
        order = {"零样本（不微调）": 0, "少样本": 1, "全量": 2}
        for setting, model, data, init, cells in sorted(rows, key=lambda x: (order[x[0]], x[1], x[3])):
            out.append(f"| {setting} | {model} | {data} | {init} | " + " | ".join(cells) + " |")

    # ---------------- 公开数据 ----------------
    for ds, title in [("deepcrack", "DeepCrack"), ("ozgenel", "高分辨率混凝土墙面（Özgenel，切片推理）")]:
        if any(k[0] == ds for k in groups):
            out.append(f"\n## 公开数据：{title}\n")
            out += ablation_table(groups, ds)

    text = "\n".join(out) + "\n"
    print(text)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
