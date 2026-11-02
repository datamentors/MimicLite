#!/bin/bash
# Pipeline check: short PPO run (default 2 GPUs, small actor, few envs) incl. checkpoint write.
# Env: A2_MOTION_DATASET (default from image), SMOKE_ENVS (1024), SMOKE_ITERS (20), MODULE (small)
set -euo pipefail
MODULE=${MODULE:-small} NUM_ENVS=${SMOKE_ENVS:-1024} ITERS=${SMOKE_ITERS:-20} \
RUN_NAME=smoke_$(date +%Y%m%d_%H%M%S) EXTRA="checkpoint_interval=10 ${EXTRA:-}" \
    bash /opt/a2/scripts/train.sh
