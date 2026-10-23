#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HOST="${1:?Usage: ./run_on_robot.sh <camera-server-host> [--show] [other options]}"
shift
exec env -u PYTHONPATH "${PY:-python}" -u run_video.py robot --host "$HOST" --port "${PORT:-5555}" "$@"
