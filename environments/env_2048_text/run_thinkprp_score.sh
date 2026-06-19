#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

set -uo pipefail
cd ${REPO_ROOT}/prime-rl
LOGS=${REPO_ROOT}/environments/env_2048_text/logs
mkdir -p "${LOGS}"
echo "================================================================"
echo "[$(date +%H:%M:%S)] Starting rl_lora_thinkprp_score (resume from step_100)"
echo "================================================================"
UV_FROZEN=1 PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/rl @ \
    ../environments/env_2048_text/configs/rl_lora_thinkprp_score.toml \
    > "${LOGS}/rl_lora_thinkprp_score.log" 2>&1
echo "[$(date +%H:%M:%S)] Finished with exit code $?"
