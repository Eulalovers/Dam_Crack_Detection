#!/usr/bin/env bash
# 消融实验：两个数据集各一个队列，在同一块 GPU 上并行运行。已完成的实验（存在 test_metrics.json）自动跳过。
cd "$(dirname "$0")/.."
RUNS=G:/Dam_crack_data/runs
mkdir -p $RUNS/logs

run() {  # dataset model loss seed
  local out="$RUNS/$1/$2_$3_s$4"
  if [ -f "$out/test_metrics.json" ]; then echo "skip $out"; return; fi
  echo "[$(date +%H:%M)] start $1 $2 $3 s$4"
  python train.py --dataset $1 --model $2 --loss $3 --seed $4 > "$RUNS/logs/$1_$2_$3_s$4.log" 2>&1
  echo "[$(date +%H:%M)] done  $1 $2 $3 s$4: $(tail -1 "$RUNS/logs/$1_$2_$3_s$4.log" | cut -c1-120)"
}

CONFIGS="unet:bce unet:dicefocal unet:combo unet_sp:combo unet_dsc:combo unet_sp_dsc:combo"

queue_deepcrack() {
  for seed in 0 1 2; do
    for c in $CONFIGS; do run deepcrack ${c%%:*} ${c##*:} $seed; done
  done
}

queue_ozgenel() {
  for c in unet:bce unet:combo unet_sp:combo unet_dsc:combo unet_sp_dsc:combo; do
    run ozgenel ${c%%:*} ${c##*:} 0
  done
}

queue_deepcrack > $RUNS/logs/queue_deepcrack.txt 2>&1 &
queue_ozgenel > $RUNS/logs/queue_ozgenel.txt 2>&1 &
wait
python scripts/summarize.py
