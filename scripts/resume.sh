#!/usr/bin/env bash
# Resume the pipeline after embed + retrieve (already done). Run from the repo root:
#   bash scripts/resume.sh 2>&1 | tee resume.log
set -uo pipefail          # no -e: one failing step does not kill the remaining ones

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NV_DATA="${NV_DATA:-$ROOT/data}"
export NV_OUT="${NV_OUT:-$ROOT/runs}"
SPLIT="${SPLIT:-test}"
echo "Resume | GPU $CUDA_VISIBLE_DEVICES | NV_OUT $NV_OUT | split $SPLIT"

# 3. reranker (now picks its weight on dev) + 4. data (noise 0.15)
python -m nv.cross_encoder
python -m nv.build_sft --n_train 6000 --n_eval 500 --noise 0.15

# 6. LoRA training first, so it starts as early as possible
python -m nv.train_lora --out ckpt/lora_mixed

# 7. fine-tuned inference
python -m nv.infer --split "$SPLIT" --fmt mcqa --name lora_mixed_mcqa --adapter ckpt/lora_mixed/final
python -m nv.infer --split "$SPLIT" --fmt qa   --name lora_mixed_qa   --adapter ckpt/lora_mixed/final

# 5. zero-shot runs whose evidence changed (zs_noev uses no evidence -> old file still valid)
python -m nv.infer --split "$SPLIT" --fmt qa   --name zs_qa
python -m nv.infer --split "$SPLIT" --fmt mcqa --name zs_mcqa

# 8. final table
python -m nv.evaluate --split "$SPLIT" --baseline zs_noev
echo "resume finished"
