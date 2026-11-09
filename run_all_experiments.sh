#!/usr/bin/env bash
set -euo pipefail

cd /data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments

PY=/home/liweixiang/anaconda3/envs/ddcolor/bin/python
LOG_DIR=/data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments/logs
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --epochs 20
  --batch-size 64
  --image-size 224
  --num-workers 8
  --lr 0.0001
  --weight-decay 0.0001
)

echo "==== $(date) cache and train resnet18_baseline ===="
$PY -m torch.distributed.run --nproc_per_node=8 train_resnet18_baseline.py "${COMMON_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/resnet18_baseline.log"

echo "==== $(date) train resnet18_frequency ===="
$PY -m torch.distributed.run --nproc_per_node=8 train_resnet18_frequency.py "${COMMON_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/resnet18_frequency.log"

echo "==== $(date) train patch_transformer ===="
$PY -m torch.distributed.run --nproc_per_node=8 train_patch_transformer.py "${COMMON_ARGS[@]}" --patch-grid 2 \
  2>&1 | tee "$LOG_DIR/patch_transformer.log"

echo "==== $(date) compare models ===="
$PY compare_three_models.py 2>&1 | tee "$LOG_DIR/model_comparison.log"

echo "==== $(date) all experiments finished ===="
