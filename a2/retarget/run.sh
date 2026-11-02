#!/usr/bin/env bash
# One-time conversion: the exact G1 SONIC references (bailinho, Ronaldo) -> AgiBot A2 clips, then the A2
# MimicLite policy tracks them in MuJoCo (headless sim2sim) and gets scored.
#   ./run.sh [clip ...]          default: bailinho Ronaldo
# Only reads the MimicLite checkout (sim2real venvs, any4hdmi/output/g1/sonic_ref, a2/ GMR tools + models)
# and the exported policy; everything it writes lands in this directory (out/, logs/, git-ignored).
# G1 sonic_ref input: sonic_ref_to_any4hdmi.py --out-dir <ML>/any4hdmi/output/g1/sonic_ref <ref dirs...>
#   A2_POLICY_DIR  exported A2 policy  (default ~/MimicLite-a2/checkpoints/a2-huge-selfcol-23000)
#   A2_MODEL       a2_serial_selfcol | a2_serial, must match the policy (default a2_serial_selfcol)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ML="${MIMICLITE_ROOT:-$HOME/MimicLite}"
A2="$(cd "$HERE/.." && pwd)"
export A2_MODEL="${A2_MODEL:-a2_serial_selfcol}"
POLICY="${A2_POLICY_DIR:-$HOME/MimicLite-a2/checkpoints/a2-huge-selfcol-23000}"
# GMR config: A2_RETARGET_CFG=<json> (e.g. cfg/xrobot_to_a2_generic_b62a773.json), else a2_sim2real.py's default;
# the one used is printed as "Use IK config:" in logs/<clip>[_<TAG>]_retarget.log
if [ -n "${A2_RETARGET_CFG:-}" ]; then export A2_GMR_CONFIG="$(realpath "$A2_RETARGET_CFG")"; else unset A2_GMR_CONFIG; fi
TAG="${TAG:-}"        # output variant: out/a2_sonic_ref_<TAG>/, logs/<clip>_<TAG>_*
SRC="$ML/any4hdmi/output/g1/sonic_ref"
OUT="$HERE/out/a2_sonic_ref${TAG:+_$TAG}"
PAD=10                # GMR IK starts cold: frame 0 is retargeted PAD extra times, then those frames are dropped
RAMP="${RAMP:-0}"     # >0: arms eased from T-pose into frame 0 over RAMP frames first (fixes wrong IK arm branch); dropped too
[ -f "$ML/env.sh" ] && source "$ML/env.sh"
for f in "$ML/sim2real/.venv/bin/python" "$ML/sim2real/venv/teleop/.venv/bin/python" "$POLICY/policy.yaml" "$SRC/manifest.json"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
CLIPS=("$@"); [ ${#CLIPS[@]} -eq 0 ] && CLIPS=(bailinho Ronaldo)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/logs/task-logs.md"; }

for c in "${CLIPS[@]}"; do
  L="$HERE/logs/${c}${TAG:+_$TAG}"
  [ -e "$SRC/motions/$c.npz" ] || { echo "missing: $SRC/motions/$c.npz" >&2; exit 1; }
  log "retarget $c (G1 sonic_ref -> A2, A2_MODEL=$A2_MODEL)"
  mkdir -p "$HERE/out/g1_padded"
  "$ML/sim2real/.venv/bin/python" "$HERE/clips.py" pad "$SRC/motions/$c.npz" "$HERE/out/g1_padded/$c.npz" "$PAD" "$RAMP" "$SRC/manifest.json"
  (cd "$ML" && sim2real/venv/teleop/.venv/bin/python "$A2/teleop/tools/check_retarget.py" \
    --motion "$HERE/out/g1_padded/$c.npz" --manifest "$SRC/manifest.json" --out "$OUT" --frames 1000000 2>&1 \
    | grep -v '^\[GMR\]' | tee "${L}_retarget.log")
  "$ML/sim2real/.venv/bin/python" "$HERE/clips.py" trim "$OUT/motions/${c}_a2.npz" "$((PAD + RAMP))"
  log "sim2sim $c (policy $POLICY)"
  (cd "$ML/sim2real" && .venv/bin/python "$A2/teleop/a2_sim2real.py" sim2real/sim_env/integrated_sim2sim.py \
    --robot a2 --policy_config "$POLICY/policy.yaml" --motion_path "$OUT/motions/${c}_a2.npz" \
    --headless --run_once --trajectory_output "${L}_traj.npz" > "${L}_sim2sim.log" 2>&1) \
    || { log "sim2sim $c FAILED, see ${L}_sim2sim.log"; exit 1; }
  "$ML/sim2real/.venv/bin/python" "$HERE/clips.py" score "${L}_traj.npz" "$OUT/motions/${c}_a2.npz" \
    | tee -a "$HERE/logs/task-logs.md"
done
