#!/usr/bin/env bash
# WS TELEOP step 3 - virtual G1 in MuJoCo (viewer URL printed, http://<workstation-ip>:808x).
# Elastic band (virtual gantry) on by default - turn it off in the viewer once the policy is following you.
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
cd sim2real
exec .venv/bin/python sim2real/sim_env/base_sim.py --robot g1
