#!/usr/bin/env bash
# A2 MimicLite checkpoint playback on the workstation (mjlab viewer on :PORT).
# usage: ./play_a2.sh [checkpoint.pt (in runs/)] [motion dataset dir (in motions/)] [extra play.py overrides...]
#   ./play_a2.sh                                   # best ckpt, wave_a2
#   ./play_a2.sh a2_small_best_checkpoint_4000.pt wave_a2 headless=true render_seconds=20
# env: MODULE (small|base|large|huge; default from ckpt name: *huge*/*large*/*base*, else small), PORT (8080), GPU (0)
set -euo pipefail
cd "$(dirname "$0")"
CKPT="${1:-a2_small_best_checkpoint_4000.pt}"; [ $# -gt 0 ] && shift
MOT="${1:-wave_a2}"; [ $# -gt 0 ] && shift
[ -f "runs/$CKPT" ] || { echo "no checkpoint: runs/$CKPT" >&2; ls runs >&2; exit 1; }
[ -f "motions/$MOT/manifest.json" ] || { echo "no motion dataset: motions/$MOT" >&2; ls motions >&2; exit 1; }
case "$CKPT" in *huge*) DEF_MODULE=huge;; *large*) DEF_MODULE=large;; *base*) DEF_MODULE=base;; *) DEF_MODULE=small;; esac
export A2_UID=$(id -u) A2_GID=$(id -g) GPUS="${GPU:-0}"
exec docker compose -f docker/compose.yaml run --rm -p "${PORT:-8080}:8080" \
  -e A2_MOTION_DATASET="/data/motions/$MOT" shell \
  bash -lc "cd /opt/aa && python projects/mimic-lite/scripts/play.py \
    task=tracking-base task/robot=a2 backend=mjlab \
    +exp=ppo/train algo/ppo/module=${MODULE:-$DEF_MODULE} \
    task.num_envs=1 task.termination.root_pos_error.enabled=false \
    checkpoint_path=/runs/$CKPT \
    hydra.run.dir=/runs/play/$(date +%Y%m%d_%H%M%S) hydra.job.chdir=true $*"
