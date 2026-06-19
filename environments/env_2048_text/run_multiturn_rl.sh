#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Real 2048 multi-turn RL: BASE+JSON first, then LoRA+JSON.
set -uo pipefail
cd ${REPO_ROOT}/prime-rl
LOGS=${REPO_ROOT}/environments/env_2048_text/logs
mkdir -p "${LOGS}"

CONFIGS=(
    rl_base_json
    rl_lora_json
)

for cfg in "${CONFIGS[@]}"; do
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Starting $cfg"
    echo "================================================================"
    UV_FROZEN=1 PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/rl @ \
        ../environments/env_2048_text/configs/${cfg}.toml \
        > "${LOGS}/${cfg}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] Finished $cfg with exit code $rc"
    sleep 30
done
echo "[$(date +%H:%M:%S)] Done"
