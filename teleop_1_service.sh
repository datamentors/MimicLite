#!/usr/bin/env bash
# WS TELEOP step 1 - XRoboToolkit PC Service (official package already installed on this workstation).
# Pico app connects to this workstation (its LAN IP is printed below). Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
R=/opt/apps/roboticsservice
if pgrep -f "$R/RoboticsServiceProcess" >/dev/null; then echo "Service already running (someone else may be using it)."; exit 0; fi
export XDG_DATA_HOME="$MIMICLITE_ROOT/.cache/xdg-data"
export LD_LIBRARY_PATH="$R:$R/lib:$R/SDK/x64:$R/SDK/x86_64:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$R/plugins/" QT_QPA_PLATFORM=offscreen
echo "XRoboToolkit service starting. Pico: connect to $(hostname -I | awk '{print $1}'). Ctrl+C to stop."
cd "$R"
./RoboticsServiceProcess 2>&1 | tee -a "$MIMICLITE_ROOT/logs/xrobo_service_$(date +%Y%m%d_%H%M%S).log"
