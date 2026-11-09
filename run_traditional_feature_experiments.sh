#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=${PYTHON:-python}
NPROC=8
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

prepare_one() {
  local feature_set="$1"
  echo "===== prepare feature cache ${feature_set} ====="
  ${PYTHON} train_resnet18_traditional_fusion.py \
    --feature-set "${feature_set}" \
    --prepare-only \
    --num-workers "${NUM_WORKERS}"
}

run_one() {
  local feature_set="$1"
  local fusion="$2"
  echo "===== start feature_set=${feature_set} fusion=${fusion} ====="
  ${PYTHON} -m torch.distributed.run --nproc_per_node=${NPROC} train_resnet18_traditional_fusion.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --feature-set "${feature_set}" \
    --fusion "${fusion}"
  echo "===== done feature_set=${feature_set} fusion=${fusion} ====="
}

# Feature extraction is CPU-side and may take longer than the default NCCL barrier timeout.
# Precompute caches in one process before launching DDP workers.
prepare_one first20
prepare_one color
prepare_one texture
prepare_one frequency
prepare_one all

# Control-variable feature screening: same backbone, same hyperparameters, concat fusion.
run_one first20 concat
run_one color concat
run_one texture concat
run_one frequency concat
run_one all concat

# Fusion-method comparison: same all-feature vector, different fusion layer.
run_one all gated
run_one all film

# Handcrafted-only reference: checks whether manual features alone carry class signal.
run_one all hand_only

${PYTHON} compare_traditional_fusion.py
