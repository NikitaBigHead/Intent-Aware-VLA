
import runpy
import sys
sys.path.insert(0, "/workspace")
import gr00t_cosmos_right13_config

sys.argv = [
    "run_gr00t_server.py",
    "--model-path", "/workspace/checkpoints/cosmos_right13_rawvideo/checkpoint-20000",
    "--embodiment-tag", "NEW_EMBODIMENT",
    "--port", "5566",
]
runpy.run_path(
    "/opt/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    run_name="__main__",
)
