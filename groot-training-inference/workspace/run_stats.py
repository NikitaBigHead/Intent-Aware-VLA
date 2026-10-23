"""Register the custom modality config, then run GR00T's stats generator."""
import runpy
import sys

sys.path.insert(0, "/workspace")
import gr00t_cosmos_right13_config  # noqa: F401  -- registers NEW_EMBODIMENT

sys.argv = [
    "stats.py",
    "--dataset-path", "/data/project/cosmos3_g1right_lerobot",
    "--embodiment-tag", "NEW_EMBODIMENT",
]
runpy.run_path("/opt/Isaac-GR00T/gr00t/data/stats.py", run_name="__main__")
