#!/bin/bash
# Host-side: PPO queue (small -> base -> huge) on GPUS for the validated dataset, logged to logs/.
# Env: GPUS (default 0,1)  MODULES (default "small base huge")
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1}
LOG=logs/queue_host_$(date +%Y%m%d_%H%M%S).log
echo "$(date -u +%FT%TZ) PPO queue ${MODULES:-small base huge} on GPUs $GPUS" | tee -a "$LOG"
GPUS=$GPUS docker compose -f docker/compose.yaml run --rm -T -e MODULES="${MODULES:-small base huge}" queue 2>&1 \
    | grep --line-buffered -E "=== |run=|Final checkpoint|\[prune\]|Error|Traceback|Killed" | tee -a "$LOG"
echo "$(date -u +%FT%TZ) queue finished" | tee -a "$LOG"
echo "- $(date -u +%H:%MZ) PPO queue finished (see $LOG)" >> logs/task-logs.md
