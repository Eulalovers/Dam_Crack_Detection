#!/usr/bin/env bash
# 大坝数据实验队列（与公开数据队列并行，单独串行运行）。已完成的实验自动跳过。
cd "$(dirname "$0")/.."
RUNS=G:/Dam_crack_data/runs
mkdir -p $RUNS/logs

run() {  # model loss seed [extra args...]  -> 输出目录名由 train.py 决定
  local model=$1 loss=$2 seed=$3 tag=$4; shift 4
  local out="$RUNS/dam/${model}_${loss}_s${seed}${tag}"
  if [ -f "$out/test_metrics.json" ]; then echo "skip $out"; return; fi
  echo "[$(date +%H:%M)] start dam $model $loss s$seed $tag"
  python train.py --dataset dam --model $model --loss $loss --seed $seed --tag "$tag" "$@" \
      > "$RUNS/logs/dam_${model}_${loss}_s${seed}${tag}.log" 2>&1
  echo "[$(date +%H:%M)] done  dam $model $loss s$seed $tag: $(tail -1 "$RUNS/logs/dam_${model}_${loss}_s${seed}${tag}.log" | cut -c1-110)"
}

zeroshot() {  # source model
  local ck="$RUNS/$1/$2_combo_s0/best.pt"
  local out="$RUNS/dam/$2_zeroshot-$1_s0"
  if [ -f "$out/test_metrics.json" ] || [ ! -f "$ck" ]; then return; fi
  # 只在源模型训练完成（test_metrics.json 存在）后评估
  [ -f "$RUNS/$1/$2_combo_s0/test_metrics.json" ] || return
  echo "[$(date +%H:%M)] zeroshot $1 $2"
  python scripts/eval_checkpoint.py --weights "$ck" --model $2 --source $1 > "$RUNS/logs/dam_zeroshot_$1_$2.log" 2>&1
}

CONFIGS="unet:bce unet:combo unet_sp:combo unet_dsc:combo unet_sp_dsc:combo"

# 1) 消融，种子 0
for c in $CONFIGS; do run ${c%%:*} ${c##*:} 0 ""; done

# 2) 零样本迁移（源模型已训练完成的才评估）
for src in deepcrack ozgenel; do for m in unet unet_sp_dsc; do zeroshot $src $m; done; done

# 3) 少样本迁移：10% 大坝数据，ImageNet 初始化 vs 高分辨率公开数据模型初始化
run unet_sp_dsc combo 0 "_frac0.1" --train_frac 0.1
while [ ! -f "$RUNS/ozgenel/unet_sp_dsc_combo_s0/test_metrics.json" ]; do sleep 120; done
run unet_sp_dsc combo 0 "_frac0.1_init-ozgenel" --train_frac 0.1 --init "$RUNS/ozgenel/unet_sp_dsc_combo_s0/best.pt"
run unet_sp_dsc combo 0 "_init-ozgenel" --init "$RUNS/ozgenel/unet_sp_dsc_combo_s0/best.pt"
for src in deepcrack ozgenel; do for m in unet unet_sp_dsc; do zeroshot $src $m; done; done

# 4) 种子 1、2
for seed in 1 2; do for c in $CONFIGS; do run ${c%%:*} ${c##*:} $seed ""; done; done

python scripts/summarize.py
