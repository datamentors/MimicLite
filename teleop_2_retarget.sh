#!/usr/bin/env bash
# WS TELEOP step 2 - sim2real pico_retarget_pub: Pico body -> GMR -> G1 pose stream (ZMQ :28701, buttons :5592).
# Viewer URL printed (open http://<workstation-ip>:808x from the Mac). Usage: ./teleop_2_retarget.sh [height_m]
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
HEIGHT="${1:-1.60}"
pgrep -f RoboticsServiceProcess >/dev/null || echo "WARNING: XRoboToolkit service not running - start ./teleop_1_service.sh first."
echo "Operator height: $HEIGHT m. Pico: whole-body tracking ON (leg trackers + calibration)."
cd sim2real
exec venv/teleop/.venv/bin/python sim2real/teleop/pico_retarget_pub.py --robot g1 --actual-human-height "$HEIGHT"
