#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PYTHON:-python}
LOG_DIR=logs
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

echo "==== $(date) train fusion_model ===="
$PY -m torch.distributed.run --nproc_per_node=8 train_fusion_model.py "${COMMON_ARGS[@]}" --patch-grid 2 \
  2>&1 | tee "$LOG_DIR/fusion_model.log"

echo "==== $(date) compare all models ===="
$PY compare_all_models.py 2>&1 | tee "$LOG_DIR/model_comparison_all.log"

echo "==== $(date) fusion experiment finished ===="
