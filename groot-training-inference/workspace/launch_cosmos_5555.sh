#!/bin/bash
cd /opt/Isaac-GR00T
source /opt/Isaac-GR00T/.venv/bin/activate
nohup python /workspace/run_server_cosmos.py > /workspace/gr00t_server.log 2>&1 &
disown
echo "PID: $!"
