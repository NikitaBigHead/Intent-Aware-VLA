
import json
import runpy
import sys
from pathlib import Path
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

CKPT = '/workspace/checkpoints/hf_cosmos_gearsonic_e07ae274/checkpoint-20000'
saved = json.loads((Path(CKPT) / "processor_config.json").read_text())
specs = saved["processor_kwargs"]["modality_configs"]["new_embodiment"]
config = {}
for name, spec in specs.items():
    spec = dict(spec)
    actions = spec.pop("action_configs", None)
    if actions is not None:
        spec["action_configs"] = [
            ActionConfig(
                rep=ActionRepresentation[a["rep"]],
                type=ActionType[a["type"]],
                format=ActionFormat[a["format"]],
                state_key=a.get("state_key"),
            )
            for a in actions
        ]
    config[name] = ModalityConfig(**spec)
register_modality_config(config)

sys.argv = [
    "run_gr00t_server.py",
    "--model-path", CKPT,
    "--embodiment-tag", "NEW_EMBODIMENT",
    "--port", "5566",
]
runpy.run_path(
    "/opt/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    run_name="__main__",
)
