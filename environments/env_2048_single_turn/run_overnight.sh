#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Overnight: main 2 (base+emprp, lora+emprp with la=0.05) then length sweep on lora
set -uo pipefail
cd ${REPO_ROOT}/prime-rl
LOGS=${REPO_ROOT}/environments/env_2048_single_turn/logs

# Note: rl_enum_lora_emprp (la=0.05) is part of the main set; sweep adds 0.0, 0.1, 0.2
CONFIGS=(
    rl_enum_base_emprp
    rl_enum_lora_emprp
    rl_enum_lora_emprp_la00
    rl_enum_lora_emprp_la01
    rl_enum_lora_emprp_la02
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
    sleep 30
done

echo "[$(date +%H:%M:%S)] All overnight runs done"
