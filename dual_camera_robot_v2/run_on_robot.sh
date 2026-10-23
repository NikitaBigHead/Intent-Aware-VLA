#!/usr/bin/env bash
# One command on the robot network:  ./run_on_robot.sh <camera-server-host> [extra run_dualcam.py options]
# Port: PORT=5555 (default). Python: PY=python (default).
# Read-only: subscribes to the robot camera server (ZMQ) and never sends commands to the robot.
set -euo pipefail
cd "$(dirname "$0")"
HOST="${1:?usage: ./run_on_robot.sh <camera-server-host> [options]}"
PORT="${PORT:-5555}"
shift
PY="${PY:-python}"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SHOW=()
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || [ "$(uname)" = Darwin ]; then SHOW=(--show); fi
exec "$PY" -B run_dualcam.py robot --host "$HOST" --port "$PORT" \
  --person-key head --hand-key ego_view \
  --log "logs/robot_${STAMP}.jsonl" --save "logs/robot_${STAMP}.mp4" --print ${SHOW[@]+"${SHOW[@]}"} "$@"
