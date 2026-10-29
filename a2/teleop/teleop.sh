#!/usr/bin/env bash
# A2 VR teleop in MuJoCo, MimicLite-style: PICO (XRoboToolkit) -> GMR -> A2 reference (ZMQ :28701)
# -> A2 MimicLite policy -> sim. One terminal per step, in this order:
#   teleop.sh service            XRoboToolkit PC Service (PICO app connects to this machine)
#   teleop.sh sim                MuJoCo A2 (viewer URL printed; elastic band on - turn it off in the viewer
#                                once the policy is following you)
#   teleop.sh pose [zero|default] browser view of A2 standing in its zero T-pose (what calibrate matches) / stand pose
#   teleop.sh calibrate [height] (once per operator) record a T-pose -> calibrated GMR config, used by retarget
#   teleop.sh retarget [height]  PICO body -> GMR -> A2 stream (operator height in m, default 1.75)
#   teleop.sh policy             A2 policy (PICO buttons: A = stand | A+B = policy on | X = follow/pause | B = hold)
# Without a headset, replace `retarget` with:
#   teleop.sh fake-pico [g1 lafan clip]   replays a G1 clip as PICO body data and presses A, A+B, X itself
#
# Uses an existing MimicLite checkout's sim2real venvs read-only (nothing is installed or changed there):
#   MIMICLITE_ROOT  checkout with sim2real/.venv and sim2real/venv/teleop/.venv   (default ~/MimicLite)
#   A2_POLICY_DIR   policy.onnx + policy.yaml exported from an A2 checkpoint      (default <a2>/../checkpoints/a2-huge-selfcol-23000)
#   XROBO_SERVICE   XRoboToolkit PC Service install dir                           (default /opt/apps/roboticsservice)
#   A2_MODEL        a2_serial_selfcol | a2_serial: sim/GMR model, match the policy (default a2_serial_selfcol)
set -euo pipefail
A2_TELEOP="$(cd "$(dirname "$0")" && pwd)"
A2_DIR="$(dirname "$A2_TELEOP")"
MIMICLITE_ROOT="${MIMICLITE_ROOT:-$HOME/MimicLite}"
A2_POLICY_DIR="${A2_POLICY_DIR:-$(dirname "$A2_DIR")/checkpoints/a2-huge-selfcol-23000}"
CALIB_DIR="${A2_CALIB_DIR:-$(dirname "$A2_DIR")/calib}"  # operator T-pose calibrations (not in git)
SIM2REAL="$MIMICLITE_ROOT/sim2real"
PY="$SIM2REAL/.venv/bin/python"
TPY="$SIM2REAL/venv/teleop/.venv/bin/python"
LAUNCH="$A2_TELEOP/a2_sim2real.py"
[ -f "$MIMICLITE_ROOT/env.sh" ] && source "$MIMICLITE_ROOT/env.sh"

need() { [ -e "$1" ] || { echo "missing: $1 ($2)" >&2; exit 1; }; }
running() { pgrep -f "$1" >/dev/null; }

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  service)
    R="${XROBO_SERVICE:-/opt/apps/roboticsservice}"
    need "$R/RoboticsServiceProcess" "XRoboToolkit PC Service"
    if running "$R/RoboticsServiceProcess"; then echo "Service already running (someone else may be using it)."; exit 0; fi
    export XDG_DATA_HOME="$A2_DIR/../logs/xdg-data" QT_QPA_PLATFORM=offscreen QT_PLUGIN_PATH="$R/plugins/"
    export LD_LIBRARY_PATH="$R:$R/lib:$R/SDK/x64:$R/SDK/x86_64:${LD_LIBRARY_PATH:-}"
    echo "XRoboToolkit service starting - PICO app: connect to this machine's IP. Ctrl+C to stop."
    cd "$R" && exec ./RoboticsServiceProcess
    ;;
  sim)
    need "$PY" "sim2real venv"
    cd "$SIM2REAL" && exec "$PY" "$LAUNCH" sim2real/sim_env/base_sim.py --robot a2 "$@"
    ;;
  pose)
    need "$PY" "sim2real venv"
    cd "$SIM2REAL" && exec "$PY" "$A2_TELEOP/tools/view_pose.py" "${1:-zero}"
    ;;
  calibrate)
    need "$TPY" "sim2real teleop venv"
    running RoboticsServiceProcess || { echo "ABORT: start 'teleop.sh service' and connect the PICO first." >&2; exit 1; }
    HEIGHT="${1:-1.75}"; mkdir -p "$CALIB_DIR"; TS=$(date +%Y%m%d_%H%M%S)
    echo "T-pose calibration (operator height $HEIGHT m): arms straight out at shoulder height, palms down,"
    echo "legs straight, feet hip-width, facing forward, hold still."
    for i in 5 4 3 2 1; do echo "  recording in $i..."; sleep 1; done
    echo "  RECORDING - hold the T-pose (10 s)"
    (cd "$SIM2REAL" && "$TPY" "$A2_TELEOP/tools/record_pico.py" 10 "$CALIB_DIR/tpose_$TS.npz")
    echo "  done - you can relax"
    (cd "$SIM2REAL" && "$TPY" "$A2_TELEOP/tools/make_xrobot_to_a2.py" --tpose "$CALIB_DIR/tpose_$TS.npz" \
      --height "$HEIGHT" --out "$CALIB_DIR/xrobot_to_a2_tpose_$TS.json" 2>&1 | grep -v '^\[GMR\]')
    ln -sf "xrobot_to_a2_tpose_$TS.json" "$CALIB_DIR/xrobot_to_a2_tpose.json"
    echo "Active calibration: $CALIB_DIR/xrobot_to_a2_tpose.json - restart 'teleop.sh retarget $HEIGHT' to use it."
    ;;
  retarget)
    need "$TPY" "sim2real teleop venv"
    running RoboticsServiceProcess || echo "WARNING: XRoboToolkit service not running - start 'teleop.sh service' first."
    HEIGHT="${1:-1.75}"; [ $# -gt 0 ] && shift
    export A2_GMR_CONFIG="${A2_GMR_CONFIG:-$CALIB_DIR/xrobot_to_a2_tpose.json}"
    [ -e "$A2_GMR_CONFIG" ] || { echo "ABORT: no T-pose calibration ($A2_GMR_CONFIG) - run 'teleop.sh calibrate $HEIGHT' first." >&2; exit 1; }
    echo "Operator height $HEIGHT m. GMR config: $A2_GMR_CONFIG"
    echo "PICO: whole-body tracking on (leg trackers + calibration)."
    cd "$SIM2REAL" && exec "$TPY" "$LAUNCH" sim2real/teleop/pico_retarget_pub.py --robot a2 --actual-human-height "$HEIGHT" "$@"
    ;;
  fake-pico)
    need "$TPY" "sim2real teleop venv"
    CLIP="${1:-walk1_subject5}"; [ $# -gt 0 ] && shift
    G1="$MIMICLITE_ROOT/any4hdmi/output/g1/lafan"
    need "$G1/motions/$CLIP.npz" "G1 LAFAN clip"
    cd "$SIM2REAL" && exec "$TPY" "$A2_TELEOP/tools/fake_pico.py" --motion "$G1/motions/$CLIP.npz" \
      --manifest "$G1/manifest.json" -- --robot a2 --no-viewer --actual-human-height 1.8 "$@"
    ;;
  policy)
    need "$PY" "sim2real venv"; need "$A2_POLICY_DIR/policy.yaml" "export: see a2/teleop/HOW_TO_USE.md"
    running "sim_env/base_sim.py" || { echo "ABORT: start 'teleop.sh sim' first." >&2; exit 1; }
    running "pico_retarget_pub.py|fake_pico.py" || { echo "ABORT: start 'teleop.sh retarget' (or fake-pico) first." >&2; exit 1; }
    cd "$SIM2REAL" && exec "$PY" "$LAUNCH" sim2real/rl_policy/tracking.py --robot a2 --robot_io zmq \
      --inference_backend onnx-cpu --policy_config "$A2_POLICY_DIR/policy.yaml" --motion_backend zmq --controller pico "$@"
    ;;
  *)
    sed -n '2,18p' "$0"; exit 1
    ;;
esac
