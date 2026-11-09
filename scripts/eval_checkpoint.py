"""零样本迁移评估：用公开混凝土数据训练好的模型，不做任何微调，直接在大坝测试集上评估。
回答"公开数据训练的模型能否直接用于大坝"这个问题。

  python scripts/eval_checkpoint.py --weights G:/Dam_crack_data/runs/ozgenel/unet_sp_dsc_combo_s0/best.pt \
      --model unet_sp_dsc --source ozgenel
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackseg.config import RUNS_DIR  # noqa: E402
from crackseg.datasets import dam_splits  # noqa: E402
from crackseg.models import build_model  # noqa: E402
from train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="训练所用的公开数据集名")
    a = ap.parse_args()
    model = build_model(a.model, pretrained=False).cuda()
    model.load_state_dict(torch.load(a.weights, map_location="cuda"))
    _, _, test = dam_splits()
    res = evaluate(model, test, "dam", topo=True)
    res = {**res["All"], "subsets": res}
    res.update(dataset="dam", model=a.model, loss=f"zeroshot-{a.source}", seed=0, init=a.weights, train_frac=0.0,
               train_minutes=0)
    out = RUNS_DIR / "dam" / f"{a.model}_zeroshot-{a.source}_s0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in res.items() if isinstance(v, float)}))


if __name__ == "__main__":
    main()
