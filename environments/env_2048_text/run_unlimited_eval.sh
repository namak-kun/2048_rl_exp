#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Diagnostic eval driver: for each ckpt, start vLLM, run unlimited eval, kill vLLM.
set -uo pipefail

cd ${REPO_ROOT}
RESULTS=${REPO_ROOT}/environments/env_2048_text/eval_unlimited_results
mkdir -p "$RESULTS"

declare -A CKPTS=(
    ["lora_seed"]="${REPO_ROOT}/environments/env_2048_single_turn/rl_enum_lora_prp/weights/step_200"
    ["thinkprp_step100"]="${REPO_ROOT}/environments/env_2048_text/rl_lora_thinkprp/weights/step_100"
    ["thinkprp_step125"]="${REPO_ROOT}/environments/env_2048_text/rl_lora_thinkprp/weights/step_125"
)

VLLM_BIN=${REPO_ROOT}/prime-rl/.venv/bin/vllm
EVAL_BIN=${REPO_ROOT}/prime-rl/.venv/bin/python

for name in lora_seed thinkprp_step100 thinkprp_step125; do
    model="${CKPTS[$name]}"
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Launching vLLM for $name"
    echo "  model=$model"
    echo "================================================================"

    # Launch vLLM in the background (use 2 GPUs via DP)
    CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM_BIN" serve "$model" \
        --port 8000 --host 127.0.0.1 \
        --data-parallel-size 2 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 8192 \
        --dtype bfloat16 \
        > "$RESULTS/${name}_vllm.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    echo "[$(date +%H:%M:%S)] vLLM PID: $VLLM_PID (logs: $RESULTS/${name}_vllm.log)"

    # Wait for vLLM to be ready
    for i in {1..120}; do
        sleep 5
        if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] vLLM is ready (after ${i}*5s)"
            break
        fi
        if (( i == 120 )); then
            echo "[$(date +%H:%M:%S)] vLLM did not start within 10 minutes; skipping"
            kill -9 $VLLM_PID 2>/dev/null
            continue 2
        fi
    done

    # Run the eval (uses GPUs 0,1 via vLLM; nothing on 2,3)
    echo "[$(date +%H:%M:%S)] Running eval"
    "$EVAL_BIN" ${REPO_ROOT}/environments/env_2048_text/eval_unlimited.py \
        --model "$model" \
        --n_games 16 \
        --max_moves 1000 \
        --max_invalid_moves 5 \
        --grid_size 4 --target_tile 2048 \
        --temperature 0.7 --max_tokens 2048 \
        --max_concurrent 8 \
        --out_json "$RESULTS/${name}.json" 2>&1 | tee "$RESULTS/${name}_eval.log"

    echo "[$(date +%H:%M:%S)] Killing vLLM PID $VLLM_PID"
    kill $VLLM_PID 2>/dev/null
    sleep 5
    kill -9 $VLLM_PID 2>/dev/null
    sleep 10  # give GPUs time to free
done

echo "================================================================"
echo "[$(date +%H:%M:%S)] All done"
echo "================================================================"
