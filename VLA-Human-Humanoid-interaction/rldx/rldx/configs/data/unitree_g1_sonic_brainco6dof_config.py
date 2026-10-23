# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom modality config for the Unitree G1 + BrainCo Revo2 (6-DOF hand)
SONIC-latent fine-tune dataset (sep-10-2026-cleaned-merged).

Distinct from the repo's built-in ``unitree_g1`` entry in
``embodiment_configs.py``: that entry assumes raw 30-step joint-target
chunks with 7-DOF hands (GR00T-WholeBodyControl's generic 3-finger hand
template) and predicts ``action.wbc``-style joint commands directly.

This dataset instead supervises on SONIC's 64-dim continuous latent
(``action.motion_token``, produced live by the SONIC encoder during
teleop) plus the two 6-DOF BrainCo Revo2 hand commands. The SONIC
decoder (``model_decoder.onnx``, run separately at 50 Hz on the robot)
turns the latent back into joint commands; ``action.wbc`` in this
dataset is a debug/reference field only and is intentionally NOT part
of the training target here.

Action target = 64 (motion_token) + 6 (left_hand_joints) + 6
(right_hand_joints) = 76-dim, chunked over ``action_horizon`` steps
(pass ``--action-horizon 40`` at the CLI — the modality config's
``delta_indices`` length below must always match that flag, see
``_validate_action_horizon_matches_modality`` in
``rldx/experiment/assembly.py``).

Video: only the two working camera streams are declared
(``depth_camera_rgb``, ``webcam``). ``external_view_camera`` is
intentionally omitted — ``LeRobotEpisodeLoader`` only opens/decodes the
mp4s for keys present in ``modality_keys``, so the excluded camera's
video is never touched (this dataset's checked-in copy on disk does
not even contain external_view_camera mp4 files any more).

State (41-dim, well under ``max_state_dim=64``, no config change
needed) mirrors GR00T-WholeBodyControl's ``_JOINT_GROUPS_FOR_STATE``
convention (left_leg/right_leg/waist/left_arm/left_hand/right_arm/
right_hand), matching this dataset's own ``meta/modality.json``
group boundaries exactly.

Register with ``EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF`` (added in
``rldx/data/embodiment_tags.py``) via
``--modality-config-path rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py
--embodiment-tag UNITREE_G1_SONIC_BRAINCO6DOF``.
"""

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ACTION_HORIZON = 40  # must match --action-horizon on the CLI

unitree_g1_sonic_brainco6dof = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["depth_camera_rgb", "webcam"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            # action.motion_token — SONIC continuous latent, not a physical
            # joint; no meaningful delta/relative counterpart in `state`, so
            # regress it as-is (ABSOLUTE = no delta transform applied).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # teleop.left_hand_joints — BrainCo Revo2, 6 DOF,
            # normalized_0_to_1 (binary-ish grip signal, like a gripper).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # teleop.right_hand_joints — same as left_hand_joints.
            # NOTE: in the original 535-episode dataset's episodes_stats.jsonl,
            # right_hand_joints is exactly 0.0 (min==max==0.0) across all
            # episodes / 182031 frames. Per the task owner, that is NOT an
            # artifact of the old task set alone: the right hand's own DOF
            # range is hardware-limited and cannot close into a full fist or
            # a functional handshake grip, even in newer episodes (hug/
            # handshake/fist-bump) where the right ARM does move (that
            # motion is carried by action.motion_token / observation.state's
            # right_arm group, not this field). So this field may stay a
            # narrow/near-degenerate signal even in the expanded dataset —
            # do not assume it suddenly spans a full functional range, but
            # also do not hardcode "always exactly 0.0" as a fact for the
            # new data: re-profile real min/max/std from the new
            # episodes_stats.jsonl once it lands, rather than trusting this
            # note or the old dataset's numbers.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    unitree_g1_sonic_brainco6dof, EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF
)
