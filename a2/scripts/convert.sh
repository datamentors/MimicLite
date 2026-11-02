#!/bin/bash
# BonesSeedA2 soma_uniform (S3, read-only) -> /data/motions/bonesseed_a2 (50 fps any4hdmi).
# Env: DATES=210531,... (default all)  LIMIT=N clips per date (test)  WORKERS (default nproc-2)
#      OUT (default /data/motions/bonesseed_a2)  SKIP_VALIDATE=1 (convert only)
set -euo pipefail
OUT=${OUT:-/data/motions/bonesseed_a2}
LOG=/logs/convert_$(basename "$OUT")_$(date +%Y%m%d_%H%M%S).log
mkdir -p /tmp/a2scratch
aws sts get-caller-identity >/dev/null || { echo "AWS auth failed: run 'aws sso login --profile ${AWS_PROFILE}' on the host" >&2; exit 2; }
python /opt/a2/tools/convert_bonesseed_a2.py \
    --out "$OUT" --mjcf-dir /opt/a2/a2_serial --scratch /tmp/a2scratch \
    ${DATES:+--dates "$DATES"} ${LIMIT:+--limit "$LIMIT"} ${WORKERS:+--workers "$WORKERS"} \
    2>&1 | tee "$LOG"
# SKIP_VALIDATE=1: stop here (the loader filter / validation build the full FK cache on disk).
if [ "${SKIP_VALIDATE:-0}" = "1" ]; then echo "conversion done, validation skipped" | tee -a "$LOG"; exit 0; fi
# Split/drop frames that MimicLite's own loader rejects (it computes body velocities its own way),
# then drop intermediate FK caches and validate once more from a clean cache.
python /opt/a2/tools/loader_filter.py "$OUT" 2>&1 | grep -v -E "^Registered|it/s\]" | tee -a "$LOG"
find /cache/any4hdmi -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
python /opt/aa/projects/mimic-lite/mimic_lite/tasks/motion.py "$OUT" 2>&1 | tail -3 | tee -a "$LOG"
du -sh "$OUT" /cache/any4hdmi | tee -a "$LOG"
