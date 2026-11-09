"""训练 + 验证选模 + 测试集评估。

示例：
  python train.py --dataset deepcrack --model unet_sp_dsc --loss combo --seed 0
  python train.py --dataset ozgenel   --model unet        --loss bce
结果写入 G:/Dam_crack_data/runs/<dataset>/<model>_<loss>_s<seed>/
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from crackseg.config import RUNS_DIR
from crackseg.datasets import (CrackDataset, LargeImageCropDataset, dam_splits, deepcrack_splits, load_pair,
                               ozgenel_splits)
from crackseg.losses import CrackLoss
from crackseg.metrics import SegEvaluator
from crackseg.models import build_model
from crackseg.tiling import predict_full, predict_tiled


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="deepcrack", choices=["deepcrack", "ozgenel", "dam"])
    ap.add_argument("--model", default="unet")
    ap.add_argument("--loss", default="combo", choices=["bce", "dicefocal", "combo"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--init", default=None, help="用已训练模型初始化（迁移微调）")
    ap.add_argument("--train_frac", type=float, default=1.0, help="只用前 x 比例的训练数据（模拟少样本）")
    return ap.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _worker_init(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def evaluate(model, pairs, dataset, topo=True):
    """pairs 为列表时返回单组指标；为 {子集: 列表} 时分别评估，并给出合并后的 All。"""
    if isinstance(pairs, dict):
        res, ev_all = {}, SegEvaluator(topo=topo)
        for k, v in pairs.items():
            res[k] = evaluate(model, v, dataset, topo)
        res["All"] = evaluate(model, [p for v in pairs.values() for p in v], dataset, topo)
        return res
    model.eval()
    ev = SegEvaluator(topo=topo)
    for pair in pairs:
        img, gt = load_pair(pair)
        prob = predict_tiled(model, img) if dataset == "ozgenel" else predict_full(model, img)
        ev.update(prob, gt)
    model.train()
    return ev.result()


def main():
    a = parse_args()
    set_seed(a.seed)
    name = f"{a.model}_{a.loss}_s{a.seed}{a.tag}"
    out_dir = RUNS_DIR / a.dataset / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.dataset == "deepcrack":
        tr, val, test = deepcrack_splits()
        train_ds = CrackDataset(tr, train=True, crop=384, repeat=4)   # 270 张 x4 = 每 epoch 1080 个样本
        epochs = a.epochs or 60
    elif a.dataset == "dam":
        tr, val, test = dam_splits()
        epochs = a.epochs or 20
    else:
        tr, val, test = ozgenel_splits()
        train_ds = LargeImageCropDataset(tr, crop=512, samples_per_epoch=2400)
        epochs = a.epochs or 30
    if a.dataset == "dam":
        if a.train_frac < 1:  # 固定种子抽样，DamCrack 与 DamSegment 按比例保留
            tr = random.Random(123).sample(tr, int(len(tr) * a.train_frac))
        # 少样本时按样本数放大 epoch，保持总迭代次数一致，便于公平比较
        train_ds = CrackDataset(tr, train=True, crop=512, repeat=max(1, round(1 / a.train_frac)))

    dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=True, persistent_workers=a.workers > 0,
                    worker_init_fn=_worker_init)
    model = build_model(a.model).cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cuda"))
        print("init from", a.init, flush=True)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.3}, {"params": dec, "lr": a.lr}], weight_decay=1e-4)
    total = epochs * len(dl)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    crit = CrackLoss(a.loss)

    log, best, t0 = [], -1.0, time.time()
    for ep in range(1, epochs + 1):
        losses = []
        for x, y, _ in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = dict(epoch=ep, loss=float(np.mean(losses)), time=round(time.time() - t0))
        if ep % a.eval_every == 0 or ep == epochs:
            m = evaluate(model, val[:600] if a.dataset == "dam" else val, a.dataset, topo=False)
            rec.update({f"val_{k}": v for k, v in m.items()})
            if m["ODS"] > best:
                best = m["ODS"]
                torch.save(model.state_dict(), out_dir / "best.pt")
        log.append(rec)
        print(json.dumps(rec), flush=True)
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cuda"))
    t1 = time.time()
    res = evaluate(model, test, a.dataset, topo=True)
    if isinstance(res, dict) and "All" in res:
        res = {**res["All"], "subsets": res}
    res.update(init=a.init, train_frac=a.train_frac, dataset=a.dataset, model=a.model, loss=a.loss, seed=a.seed, best_val_ODS=best,
               train_minutes=round((t1 - t0) / 60, 1), test_seconds=round(time.time() - t1, 1),
               params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print("TEST", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
