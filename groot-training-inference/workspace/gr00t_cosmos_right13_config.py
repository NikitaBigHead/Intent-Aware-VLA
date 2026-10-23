"""GR00T modality for the Cosmos3 + RoboHaMeR right-arm dataset.

Action is 13 absolute channels: right arm 7 joints in radians, then the six
BrainCo hand motors normalized to [0, 1]. Left side, legs, waist, the mimic
*_distal joints and the constant *_tip joints are all excluded at conversion
time -- see convert_cosmos_to_lerobot.py.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joint_positions"],
        action_configs=[ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}
register_modality_config(config)
