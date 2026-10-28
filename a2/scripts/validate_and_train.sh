#!/bin/bash
# Host-side chain (run from ~/a2-mimiclite with setsid nohup) for an already converted dataset:
# loader filter (one full FK-cache build, then subset re-checks) -> final validation (builds the
# cache training reuses) -> PPO small -> base -> huge on GPUS.
# Env: GPUS (default 0,1)  MODULES (default "small base huge")
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1}
LOG=logs/validate_and_train_$(date +%Y%m%d_%H%M%S).log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "loader filter + validation"
# FK-cache build on GPU 0 (default is CPU: ~80 min for the full dataset on AMAX's 8-core EPYC).
GPUS=0 docker compose -f docker/compose.yaml run --rm -T -e ANY4HDMI_CACHE_BUILD_DEVICE=cuda shell bash -c '
    set -euo pipefail
    OUT=/data/motions/bonesseed_a2
    python /opt/a2/tools/loader_filter.py "$OUT" 2>&1 | grep -v -E "^Registered|it/s\]|^FK:|^Write to storage|^Interpolat"
    python /opt/aa/projects/mimic-lite/mimic_lite/tasks/motion.py "$OUT" 2>&1 | grep -E "validation|Loaded|Error|Traceback"
    du -sh "$OUT" /cache/any4hdmi' 2>&1 | tee -a "$LOG"
grep -q "Motion dataset validation passed" "$LOG" || { log "validation failed - not training"; exit 1; }
echo "- $(date -u +%H:%MZ) full dataset validated ($(grep -o 'frames=[0-9]*' "$LOG" | tail -1)); starting PPO queue on GPUs $GPUS" >> logs/task-logs.md

log "PPO queue ${MODULES:-small base huge} on GPUs $GPUS"
GPUS=$GPUS docker compose -f docker/compose.yaml run --rm -T -e MODULES="${MODULES:-small base huge}" queue 2>&1 \
    | grep --line-buffered -E "=== |run=|Final checkpoint|\[prune\]|Error|Traceback|Killed" | tee -a "$LOG"
log "queue finished"
echo "- $(date -u +%H:%MZ) PPO queue finished (see $LOG)" >> logs/task-logs.md
