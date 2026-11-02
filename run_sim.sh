#!/usr/bin/env bash
# MimicLite G1 sim2sim launcher (isolated env: sim2real/.venv + repo-local caches).
# usage: ./run_sim.sh <ppo|roa|v1_1> [motion_name|path.npz] [extra integrated_sim2sim args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VER="${1:?usage: $0 <ppo|roa|v1_1> [motion] [args...]}"; shift
MOTION="${1:-walk1_subject1}"; [ $# -gt 0 ] && shift
CFG="$ROOT/sim2real/checkpoints/mimiclite-g1/$VER/policy.yaml"
[ -f "$CFG" ] || { echo "no checkpoint: $CFG (expected ppo|roa|v1_1)" >&2; exit 1; }
case "$MOTION" in
  *.npz) MPATH="$MOTION" ;;
  *) MPATH="$ROOT/any4hdmi/output/g1/lafan/motions/$MOTION.npz" ;;
esac
[ -f "$MPATH" ] || { echo "no motion: $MPATH (ls $ROOT/any4hdmi/output/g1/lafan/motions)" >&2; exit 1; }
export HF_HOME="$ROOT/.cache/hf"
cd "$ROOT/sim2real"
exec .venv/bin/python sim2real/sim_env/integrated_sim2sim.py --robot g1 \
  --policy_config "$CFG" --motion_path "$MPATH" "$@"
