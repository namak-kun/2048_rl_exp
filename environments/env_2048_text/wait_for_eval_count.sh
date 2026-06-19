#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Wait until the orchestrator log has at least N "SUCCESS Evaluated" lines.
set -uo pipefail
LOG="${REPO_ROOT}/environments/env_2048_text/rl_lora_thinkprp/logs/orchestrator.stdout"
TARGET="${1:-12}"  # step 0 (4) + step 25 (4) + step 50 (4) = 12
PROCESS_NAME="rl_lora_thinkprp"
POLL_SEC="${2:-120}"

echo "[$(date '+%H:%M:%S')] Watchdog started; waiting for ${TARGET} eval lines."
while true; do
    if [[ -f "$LOG" ]]; then
        n=$(sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" | tr '\r' '\n' | grep -c "SUCCESS Evaluated" || true)
        echo "[$(date '+%H:%M:%S')] eval lines: $n / $TARGET"
        if (( n >= TARGET )); then
            echo "[$(date '+%H:%M:%S')] Target reached. Exiting."
            exit 0
        fi
    else
        echo "[$(date '+%H:%M:%S')] Log not yet present"
    fi
    if ! pgrep -f "$PROCESS_NAME" >/dev/null; then
        echo "[$(date '+%H:%M:%S')] Process gone; exiting."
        exit 2
    fi
    sleep "$POLL_SEC"
done
