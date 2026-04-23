#!/bin/bash
# Eval checkpoints using vf-eval on multiple grid sizes
# Run from prime-rl directory: ../environments/env_2048_single_turn/eval_ckpt.sh [step_40|all]
# Results saved to: environments/env_2048_single_turn/outputs/evals/

set -e

CKPT_BASE="../environments/env_2048_single_turn/rl_outputs_new_data/weights"

# Determine which checkpoints to eval
if [ "$1" = "all" ] || [ -z "$1" ]; then
    CHECKPOINTS=$(ls "$CKPT_BASE" | sort -t_ -k2 -n)
else
    CHECKPOINTS="$1"
fi

# Verify checkpoints exist
for CHECKPOINT in $CHECKPOINTS; do
    if [ ! -d "$CKPT_BASE/$CHECKPOINT" ]; then
        echo "Checkpoint not found: $CKPT_BASE/$CHECKPOINT"
        echo "Available:"
        ls "$CKPT_BASE"
        exit 1
    fi
done

echo "=== Will eval: $CHECKPOINTS ==="

for CHECKPOINT in $CHECKPOINTS; do
    CKPT_DIR="$CKPT_BASE/$CHECKPOINT"
    
    echo ""
    echo "========================================"
    echo "=== Checkpoint: $CHECKPOINT ==="
    echo "========================================"

    # Start inference
    echo "Starting inference server..."
    uv run inference --model.name "$CKPT_DIR" &
    INFER_PID=$!

    # Wait for server
    echo "Waiting for server..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        sleep 2
    done

    # Run evals on multiple grid sizes
    for GRID in 4 5 6; do
        echo ""
        echo "=== ${GRID}x${GRID} Grid ==="
        uv run vf-eval env-2048-single-turn \
            -m "$CKPT_DIR" \
            -b http://localhost:8000/v1 \
            -n 1000 \
            -r 1 \
            --max-tokens 256 \
            --env-dir-path ../environments \
            -a "{\"balanced_difficulty\": true, \"grid_size\": $GRID, \"num_eval_examples\": 1000, \"num_train_examples\": 1000, \"min_moves\": 0, \"max_moves\": 200}" \
            -s
    done

    # Stop server before next checkpoint
    echo "Stopping server..."
    kill $INFER_PID 2>/dev/null || true
    wait $INFER_PID 2>/dev/null || true
done

echo ""
echo "Results saved to: ../environments/env_2048_single_turn/outputs/evals/"
echo "Done!"
