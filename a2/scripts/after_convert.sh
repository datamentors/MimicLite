#!/bin/bash
# Host-side chain (run from ~/a2-mimiclite with nohup): wait for the a2_convert container to finish,
# validate the dataset with MimicLite's loader, then train PPO small -> base -> huge on GPUS.
# Env: GPUS (default 0,1)  MODULES (default "small base huge")
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1}
LOG=logs/after_convert_$(date +%Y%m%d_%H%M%S).log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "waiting for a2_convert"
code=$(docker wait a2_convert)
log "a2_convert exited with $code"
docker logs a2_convert 2>&1 | grep -E "^\[|done:|Error|Traceback" | tail -3 | tee -a "$LOG"
[ "$code" = "0" ] || { log "conversion failed - stopping (rerun the convert service to resume)"; exit 1; }
docker rm a2_convert >/dev/null

log "loader filter + validation"
GPUS=none docker compose -f docker/compose.yaml run --rm -T shell bash -c '
    set -euo pipefail
    OUT=/data/motions/bonesseed_a2
    python /opt/a2/tools/loader_filter.py "$OUT" 2>&1 | grep -v -E "^Registered|it/s\]"
    find /cache/any4hdmi -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    python /opt/aa/projects/mimic-lite/mimic_lite/tasks/motion.py "$OUT" 2>&1 | tail -3
    du -sh "$OUT" /cache/any4hdmi' 2>&1 | tee -a "$LOG"
grep -q "Motion dataset validation passed" "$LOG" || { log "validation failed - not training"; exit 1; }
echo "- $(date -u +%H:%MZ) full dataset converted + validated ($(grep -o 'frames=[0-9]*' "$LOG" | tail -1)); starting PPO queue on GPUs $GPUS" >> logs/task-logs.md

log "PPO queue ${MODULES:-small base huge} on GPUs $GPUS"
GPUS=$GPUS docker compose -f docker/compose.yaml run --rm -T -e MODULES="${MODULES:-small base huge}" queue 2>&1 \
    | grep --line-buffered -E "=== |run=|Final checkpoint|\[prune\]|Error|Traceback|Killed" | tee -a "$LOG"
log "queue finished"
echo "- $(date -u +%H:%MZ) PPO queue finished (see $LOG)" >> logs/task-logs.md
