#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

set -uo pipefail
cd ${REPO_ROOT}/environments/env_2048_text
OUT=${REPO_ROOT}/environments/env_2048_text/opsd_runs/base_v1
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] Starting OPSD from base Qwen3"
CUDA_VISIBLE_DEVICES=0 ${REPO_ROOT}/prime-rl/.venv/bin/python train_opsd_2048.py \
  --model PrimeIntellect/Qwen3-0.6B \
  --output_dir "$OUT" \
  --num_steps 100 \
  --batch_size 4 \
  --max_new_tokens 1024 \
  --lora_rank 32 --lora_alpha 64 \
  --lr 5e-5 \
  --max_grad_norm 0.1 \
  --jsd_token_clip 0.05 \
  --oracle_depth 3 \
  --save_every 25 \
  --beta 0 \
  > "$OUT/train.log" 2>&1
echo "[$(date +%H:%M:%S)] Done with exit $?"
