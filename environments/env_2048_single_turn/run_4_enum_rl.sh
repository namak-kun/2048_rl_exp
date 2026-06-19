#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Sequentially run all 4 enumerate-task RL experiments.
# Each uses 4 GPUs (2 train, 2 inference). Logs to logs/.

set -uo pipefail
cd ${REPO_ROOT}/prime-rl
LOGS=${REPO_ROOT}/environments/env_2048_single_turn/logs

CONFIGS=(
    rl_enum_base_prp
    rl_enum_lora_prp
    rl_enum_base_em
    rl_enum_lora_em
)

for cfg in "${CONFIGS[@]}"; do
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Starting $cfg"
    echo "================================================================"
    UV_FROZEN=1 PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/rl @ \
        ../environments/env_2048_single_turn/configs/${cfg}.toml \
        > "${LOGS}/${cfg}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] Finished $cfg with exit code $rc"
    # Always give vLLM/workers time to release GPU memory before next run
    sleep 30
done

echo "================================================================"
echo "[$(date +%H:%M:%S)] All 4 runs done"
echo "================================================================"
