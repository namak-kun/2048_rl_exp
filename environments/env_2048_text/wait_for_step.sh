#!/usr/bin/env bash
# REPO_ROOT = path to the repo root containing environments/ and prime-rl/
# Override via env, or derive from script location
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Watchdog: polls the run's orchestrator log and exits when target_step has completed.
# Also exits if the run is no longer running.
set -uo pipefail

LOG="${REPO_ROOT}/environments/env_2048_text/rl_lora_thinkprp/logs/orchestrator.stdout"
TARGET="${1:-25}"
PROCESS_NAME="rl_lora_thinkprp"  # what to grep ps for to detect run death
POLL_SEC="${2:-60}"

echo "[$(date '+%H:%M:%S')] Watchdog started; waiting for Step ${TARGET} or run exit. Polling every ${POLL_SEC}s."

while true; do
    if [[ -f "$LOG" ]]; then
        # Strip ANSI + carriage returns then look for "SUCCESS Step <N> |" where N >= TARGET
        latest=$(sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" | tr '\r' '\n' \
            | grep -oE 'SUCCESS Step [0-9]+ \|' | tail -1 | grep -oE '[0-9]+' || true)
        if [[ -n "$latest" ]] && (( latest >= TARGET )); then
            echo "[$(date '+%H:%M:%S')] Reached step ${latest} (target ${TARGET}). Exiting."
            exit 0
        fi
        latest_eval=$(sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" | tr '\r' '\n' \
            | grep "SUCCESS Evaluated" | tail -1 || true)
        echo "[$(date '+%H:%M:%S')] latest_step=${latest:-none}; latest_eval: ${latest_eval:0:120}"
    else
        echo "[$(date '+%H:%M:%S')] Log not yet present: $LOG"
    fi

    # Detect run death
    if ! pgrep -f "$PROCESS_NAME" >/dev/null; then
        echo "[$(date '+%H:%M:%S')] Process matching '$PROCESS_NAME' is gone; exiting."
        exit 2
    fi

    sleep "$POLL_SEC"
done
