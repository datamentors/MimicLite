#!/bin/bash
# One-off (2026-10-27, user: skip Base): wait until the running queue finishes PPO small
# (train.sh exited incl. final prune), stop the queue before Base trains, then run PPO huge on 0,1.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/skip_to_huge_$(date +%Y%m%d_%H%M%S).log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
QLOG=$(ls -t runs/queue_*.log | head -1)
log "watching $QLOG for 'done PPO small'"
until grep -q "done PPO small" "$QLOG"; do sleep 5; done
log "small done; stopping queue before base"
pkill -f "bash scripts/queue_host.sh" || true
for id in $(docker ps --filter ancestor=a2-mimiclite:latest -q); do docker stop -t 30 "$id" >/dev/null; done
sleep 5
B=$(ls -td runs/a2_ppo_base_* 2>/dev/null | head -1 || true)
[ -n "$B" ] && log "aborted base run dir left for review: $B"
echo "- $(date -u +%H:%MZ) PPO small done; Base skipped (user); starting PPO huge on GPUs 0,1" >> logs/task-logs.md
log "starting huge"
MODULES=huge GPUS=0,1 bash scripts/queue_host.sh 2>&1 | tee -a "$LOG"
