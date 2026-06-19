#!/usr/bin/env bash
# Evaluate a checkpoint (base or LoRA-merged) on the single-turn 2048 task
# using vf-eval against a vLLM server. This matches the parser/reward path
# that prime-rl RL will see.
#
# Env vars:
#   PORT     server port (default 8123) — set unique per parallel instance
#   GPU      CUDA device(s), e.g. "0" or "0,1" (default 0)
#
# Usage:
#   ./eval_lora.sh <checkpoint_dir_or_hf_id> <output_format: xml|json> [<num_examples>]
#
# Examples:
#   PORT=8123 GPU=0 ./eval_lora.sh lora_outputs_xml/merged xml 200
#   PORT=8124 GPU=1 ./eval_lora.sh lora_outputs_json/merged json 200
#   PORT=8125 GPU=2 ./eval_lora.sh Qwen/Qwen3-0.6B xml 200
#
# Results are saved by vf-eval under the env's outputs/evals/ directory.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ckpt_dir_or_hf_id> <output_format: xml|json> [<num_examples>]"
    exit 1
fi

MODEL="$1"
FORMAT="$2"
NUM="${3:-200}"

if [ -d "$MODEL" ]; then
    MODEL="$(readlink -f "$MODEL")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIME_RL="$(cd "$SCRIPT_DIR/../../prime-rl" && pwd)"

if [ "$FORMAT" != "xml" ] && [ "$FORMAT" != "json" ]; then
    echo "format must be 'xml' or 'json'"
    exit 1
fi

PORT="${PORT:-8123}"
GPU="${GPU:-0}"
VENV="$PRIME_RL/.venv/bin"

mkdir -p "$SCRIPT_DIR/logs"
TAG="$(basename "$MODEL")-$FORMAT-port$PORT"
SERVER_LOG="$SCRIPT_DIR/logs/server_${TAG}.log"

echo "=== Starting vLLM server: model=$MODEL port=$PORT gpu=$GPU ==="
CUDA_VISIBLE_DEVICES="$GPU" "$VENV/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.4}" \
    --max-model-len 6144 \
    --reasoning-parser qwen3 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID  log: $SERVER_LOG"

cleanup() {
    echo "Stopping server $SERVER_PID..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Server up after ${i} attempts ($((i * 2))s)"
        break
    fi
    sleep 2
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Server failed to start. Last 60 log lines:"
    tail -60 "$SERVER_LOG"
    exit 1
fi

run_eval() {
    local think="$1"
    local grid="$2"
    local tag="${FORMAT}-think${think}-${grid}x${grid}"
    local max_tokens=4096
    if [ "$think" = "false" ]; then
        max_tokens=128
    fi
    echo ""
    echo "=== Eval $tag (max_tokens=$max_tokens) ==="
    local env_args
    env_args="{\"balanced_difficulty\": true, \"grid_size\": ${grid}, \"num_eval_examples\": ${NUM}, \"num_train_examples\": 50, \"output_format\": \"${FORMAT}\"}"

    local sampling_args
    sampling_args="{\"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": ${think}}}}"

    OPENAI_API_KEY=EMPTY "$VENV/vf-eval" env-2048-single-turn \
        -m "$MODEL" \
        --api-base-url "http://127.0.0.1:${PORT}/v1" \
        --api-key-var OPENAI_API_KEY \
        -n "$NUM" \
        -r 1 \
        --max-tokens "$max_tokens" \
        --env-dir-path "$ENVS_DIR" \
        -a "$env_args" \
        --sampling-args "$sampling_args" \
        --save-results 2>&1 | tail -25 || true
}

for grid in 4 5 6; do
    for think in false true; do
        run_eval "$think" "$grid"
    done
done

echo "Done."
