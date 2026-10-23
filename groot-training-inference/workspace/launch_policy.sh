#!/bin/bash
docker exec -it tactile-train bash -lc "source /opt/Isaac-GR00T/.venv/bin/activate && cd /opt/Isaac-GR00T && python /workspace/run_server_cosmos.py"
