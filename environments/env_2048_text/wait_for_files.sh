#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

set -uo pipefail
DIR=${REPO_ROOT}/environments/env_2048_text/eval_unlimited_results
TARGETS=(lora_seed.json thinkprp_step100.json thinkprp_step125.json)
POLL=120

echo "[$(date '+%H:%M:%S')] Watchdog: waiting for ${#TARGETS[@]} result files in $DIR"
while true; do
    n=0
    for f in "${TARGETS[@]}"; do
        [[ -f "$DIR/$f" ]] && n=$((n+1))
    done
    echo "[$(date '+%H:%M:%S')] $n / ${#TARGETS[@]} done"
    if (( n >= ${#TARGETS[@]} )); then
        echo "[$(date '+%H:%M:%S')] All done"
        exit 0
    fi
    if ! pgrep -f "run_unlimited_eval" >/dev/null; then
        echo "[$(date '+%H:%M:%S')] Driver gone, exiting"
        exit 2
    fi
    sleep $POLL
done
