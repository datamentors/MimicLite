#!/usr/bin/env bash
# WS TELEOP step 4 - MimicLite policy following the live Pico stream, driving the MuJoCo G1 from step 3.
# Usage: ./teleop_4_policy.sh [roa|ppo]  (default roa; ~0.12 s stream delay)
# v1_1 does NOT work on the live stream: it needs left/right_palm_link, which sim2real's live G1 stream lacks.
# Pico buttons: A = stand pose | A+B = policy ON | X = start/pause following you | B = hold
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
VER="${1:-roa}"
if [ "$VER" = "v1_1" ]; then echo "v1_1 cannot run on live teleop (needs palm bodies the live stream lacks). Use roa or ppo."; exit 1; fi
pgrep -f "sim_env/base_sim.py" >/dev/null || { echo "ABORT: start ./teleop_3_sim.sh first."; exit 1; }
pgrep -f "teleop/pico_retarget_pub.py" >/dev/null || { echo "ABORT: start ./teleop_2_retarget.sh first."; exit 1; }
cd sim2real
exec .venv/bin/python sim2real/rl_policy/tracking.py --robot g1 --robot_io zmq --inference_backend onnx-cpu \
  --policy_config "checkpoints/mimiclite-g1/$VER/policy.yaml" --motion_backend zmq --controller pico
