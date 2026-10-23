"""GR00T modality for the synthetic G1+BrainCo RoboHaMeR trajectories.

The action is 46 MuJoCo joint positions: left arm 7, left fingers 16,
right arm 7, right fingers 16. This is deliberately separate from the
real robot's 64-D WBC motion-token interface.
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
