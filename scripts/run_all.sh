#!/usr/bin/env bash
# NewsVQA-lite full pipeline
# Run from the repo root.

set -euo pipefail

# Project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# GPU 0 by default (Colab has one); set CUDA_VISIBLE_DEVICES to pick another
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

# Data and outputs inside the repo (both git-ignored)
export NV_DATA="${NV_DATA:-$ROOT/data}"
export NV_OUT="${NV_OUT:-$ROOT/runs}"

# Choose dev or test
SPLIT="${SPLIT:-test}"

echo "======================================"
echo "NewsVQA Pipeline"
echo "Project root: $ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "NV_DATA: $NV_DATA"
echo "NV_OUT: $NV_OUT"
echo "Split: $SPLIT"
echo "======================================"

# Validate directories
if [[ ! -d "$NV_DATA/TARA_dataset" ]]; then
    echo "ERROR: TARA_dataset not found in $NV_DATA"
    exit 1
fi

if [[ ! -d "$NV_OUT/prepared" ]]; then
    echo "ERROR: prepared corpus not found in $NV_OUT"
    exit 1
fi

if [[ ! -d "$NV_OUT/geonames" ]]; then
    echo "ERROR: geonames directory not found in $NV_OUT"
    exit 1
fi

# 1. CLIP embeddings
python -m nv.embed

# 2. Article retrieval
python -m nv.retrieve

# 3. Cross-encoder reranking
python -m nv.cross_encoder

# 4. Build training/evaluation data
python -m nv.build_sft --n_train 6000 --n_eval 500 --noise 0.15

# 5. Zero-shot baselines
python -m nv.infer --split "$SPLIT" --fmt qa_noev --name zs_noev
python -m nv.infer --split "$SPLIT" --fmt qa --name zs_qa
python -m nv.infer --split "$SPLIT" --fmt mcqa --name zs_mcqa

python -m nv.evaluate --split "$SPLIT" --baseline zs_noev

# 6. LoRA training
python -m nv.train_lora --out ckpt/lora_mixed

# 7. LoRA inference
python -m nv.infer --split "$SPLIT" --fmt mcqa \
    --name lora_mixed_mcqa --adapter ckpt/lora_mixed/final

python -m nv.infer --split "$SPLIT" --fmt qa \
    --name lora_mixed_qa --adapter ckpt/lora_mixed/final

# 8. Final evaluation
python -m nv.evaluate --split "$SPLIT" --baseline zs_noev

echo "======================================"
echo "NewsVQA pipeline completed!"
echo "======================================"

