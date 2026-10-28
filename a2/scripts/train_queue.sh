#!/bin/bash
# Run PPO small -> base -> huge one after another (stops at the first failure).
# Env: MODULES (default "small base huge"), plus everything train.sh reads.
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
for m in ${MODULES:-small base huge}; do
    echo "=== $(date -u +%FT%TZ) start PPO $m" | tee -a /runs/queue_$TS.log
    MODULE=$m RUN_NAME=a2_ppo_${m}_$TS bash /opt/a2/scripts/train.sh
    echo "=== $(date -u +%FT%TZ) done PPO $m" | tee -a /runs/queue_$TS.log
done
