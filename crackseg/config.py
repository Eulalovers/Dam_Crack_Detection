"""全局路径配置。代码放在 D 盘，数据和训练产物放在 G 盘（D 盘空间不足）。"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CRACK_DATA_ROOT", "G:/Dam_crack_data"))
RAW_DIR = DATA_ROOT / "raw"
RUNS_DIR = DATA_ROOT / "runs"          # 训练日志、权重
OUTPUT_DIR = DATA_ROOT / "outputs"     # 推理结果、报告

DEEPCRACK_DIR = RAW_DIR / "deepcrack"
OZGENEL_DIR = RAW_DIR / "ozgenel"

# 大坝混凝土裂缝按缝宽分级（单位 mm）。
# 参考 SL 713-2015《水工混凝土结构缺陷检测技术规程》及大坝混凝土裂缝分类表：
#   A 类: δ < 0.2   B 类: 0.2 ≤ δ < 0.3   C 类: 0.3 ≤ δ < 0.4   D 类: δ ≥ 0.4
# 规范中分类同时考虑缝深，图像法只能获得表面缝宽，因此这里仅按缝宽给出“初判等级”。
# 使用前请对照规范原文复核阈值。
CRACK_GRADE_THRESHOLDS_MM = [(0.2, "A"), (0.3, "B"), (0.4, "C"), (float("inf"), "D")]
