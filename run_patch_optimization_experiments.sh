#!/usr/bin/env bash
set -euo pipefail

cd /data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments

PYTHON=/home/liweixiang/anaconda3/envs/ddcolor/bin/python
NPROC=8
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}

run_patch() {
  local name="$1"
  shift
  echo "===== start ${name} ====="
  ${PYTHON} -m torch.distributed.run --nproc_per_node=${NPROC} train_patch_transformer_optimized.py \
    --mode patch \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --experiment-name "${name}" \
    "$@"
  echo "===== done ${name} ====="
}

run_fusion() {
  local fusion="$1"
  local name="patch_opt_frequency_${fusion}"
  echo "===== start ${name} ====="
  ${PYTHON} -m torch.distributed.run --nproc_per_node=${NPROC} train_patch_transformer_optimized.py \
    --mode fusion \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --feature-set frequency \
    --fusion "${fusion}" \
    --patch-grid 3 \
    --transformer-layers 2 \
    --transformer-heads 8 \
    --use-cls \
    --experiment-name "${name}"
  echo "===== done ${name} ====="
}

# Patch Transformer controlled optimization.
run_patch patch_opt_g2_cls_global --patch-grid 2 --transformer-layers 2 --transformer-heads 8 --use-cls
run_patch patch_opt_g3_cls_global --patch-grid 3 --transformer-layers 2 --transformer-heads 8 --use-cls
run_patch patch_opt_g3_cls_noglobal --patch-grid 3 --transformer-layers 2 --transformer-heads 8 --use-cls --no-global

# Fusion comparison with the selected frequency handcrafted features.
run_fusion concat
run_fusion gated
run_fusion film
run_fusion crossattn

${PYTHON} compare_patch_optimization.py
