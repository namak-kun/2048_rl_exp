#!/usr/bin/env bash
# Stage 1: SFT training for valid moves (no-think mode)
#
# Usage (from prime-rl directory):
#   ../environments/env_2048_single_turn/run_sft.sh [generate|train|all]
#
# Steps:
#   1. generate - Generate SFT training data
#   2. train    - Run SFT training
#   all         - Run both steps (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR"
CONFIG="$ENV_DIR/configs/sft.toml"
DATA_DIR="$ENV_DIR/sft_data"

COMMAND="${1:-all}"

generate_data() {
    echo "========================================="
    echo "=== Generating SFT data ==="
    echo "========================================="
    
    # Generate 10k examples across 4x4, 5x5, and 6x6 grids
    # 60% 4x4 (main training size), 25% 5x5, 15% 6x6
    uv run python "$ENV_DIR/generate_sft_data.py" \
        --output "$DATA_DIR" \
        --num_examples 10000 \
        --grid_sizes 4 5 6 \
        --grid_weights 0.6 0.25 0.15 \
        --balanced_difficulty \
        --seed 42
    
    echo ""
    echo "Data saved to: $DATA_DIR"
}

train() {
    echo "========================================="
    echo "=== Starting SFT training ==="
    echo "========================================="
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: SFT data not found at $DATA_DIR"
        echo "Run: $0 generate"
        exit 1
    fi
    
    uv run sft @ "$CONFIG"
}

case "$COMMAND" in
    generate)
        generate_data
        ;;
    train)
        train
        ;;
    all)
        generate_data
        echo ""
        train
        ;;
    *)
        echo "Usage: $0 [generate|train|all]"
        exit 1
        ;;
esac
