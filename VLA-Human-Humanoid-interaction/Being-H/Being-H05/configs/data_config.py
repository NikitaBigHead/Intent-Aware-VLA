# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from typing import Optional
from BeingH.utils.schema import RotationType
from BeingH.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from BeingH.dataset.transform.concat import ConcatTransform
from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from BeingH.utils.constants import TARGET_STATE_ROTATION_TYPE, TARGET_ACTION_ROTATION_TYPE, TARGET_STATE_ROTATION_DIM, TARGET_ACTION_ROTATION_DIM, AGIBOT_ABS_OR_RELA


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column (exclusive)")
    absolute: bool = True

    rotation_type: Optional[RotationType] = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = Field(True, description="Whether the data is continuous (floating point)")


class BaseDataConfig(ABC):
    def __init__(self, embodiment_tag, use_fixed_view, max_view_num, 
                obs_indices=[0], action_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices
        self.action_indices = action_indices

    @abstractmethod
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """
        Define how to extract and name new modalities from raw Parquet columns.
        Returns: {'modality.key': ModalityDef(...), ...}
        """
        pass

    def get_sampling_indices(self) -> Dict[str, List[int]]:
        """Define sampling indices"""
        sampling_map = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        """
        Define a complete, ordered data transformation pipeline.
        Returns a ComposedModalityTransform object.
        """
        pass

    def add_video_modality(self, modalities):
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
            # rand_view_num = random.randint(1, len(self.VIDEO_SOURCE_COLUMNS))
            # video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), rand_view_num)
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), max_view_num)
   
        for video_key in video_keys:
            modalities[video_key] = ModalityDef(source_column=self.VIDEO_SOURCE_COLUMNS[video_key], start=0, end=0)

        return modalities


class LiberoOriginDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.top_view']
    VIDEO_SOURCE_COLUMNS = {'video.top_view': 'observation.images.image'}
    STATE_KEYS = ['state.state']
    ACTION_KEYS = ['action.action']

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {'state.state': 'min_max'} 
    action_normalization_modes = {'action.action': 'min_max'}

    state_action_type = {'state.state': "7-d absolute state (xyz,roll,pitch,yaw,pad) + 1-d gripper pos", 
                         'action.action': "6-d relative action (xyz,roll,pitch,yaw) + 1-d gripper pos"
                        }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),
            'state.state': ModalityDef(source_column='observation.state', start=0, end=8),
            'action.action': ModalityDef(source_column='action', start=0, end=7, absolute=False),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoNoNormDataConfig(LiberoOriginDataConfig):
    VIDEO_KEYS = ['video.top_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.top_view': 'observation.images.image',
        'video.wrist_view': 'observation.images.wrist_image',
    }
    STATE_KEYS = ['state.eef_position', 'state.eef_rotation', 'state.libero_gripper_position']
    ACTION_KEYS = ['action.eef_position', 'action.eef_rotation', 'action.gripper_position']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':     (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.libero_gripper_position': (44, 46),

        'action.eef_position':    (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position':(18, 19),
    }

    state_normalization_modes = {
    }
    
    action_normalization_modes = {
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.libero_gripper_position': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='observation.state', start=3, end=6, rotation_type="axis_angle"),
            'state.libero_gripper_position': ModalityDef(source_column='observation.state', start=6, end=8),

            'action.eef_position': ModalityDef(source_column='action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='action', start=6, end=7),
        }
        modalities = self.add_video_modality(modalities)

        return modalities


class RobocasaHumanDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.left_view', 'video.right_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.left_view': 'observation.images.left_view',
        'video.right_view': 'observation.images.right_view',
        'video.wrist_view': 'observation.images.wrist_view',
    }
    STATE_KEYS = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    ACTION_KEYS = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
        "action.base_motion",
        "action.control_mode",
    ]

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':  (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.gripper_qpos': (44, 46),
        'state.base_position': (70, 73),
        'state.base_rotation': (73, 76),

        'action.eef_position': (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position': (18, 19),
        'action.base_motion': (70, 74),
        'action.control_mode': (74, 75),
    }

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {} 
    # action_normalization_modes = {}

    action_normalization_modes = {
        # "action.end_effector_position": "min_max",
        # "action.end_effector_rotation": "min_max",
        "action.gripper_position": "binary",
        # "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.gripper_qpos': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='world_abs_state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='world_abs_state', start=3, end=6, rotation_type="axis_angle"),
            'state.gripper_qpos': ModalityDef(source_column='world_abs_state', start=6, end=8),
            'state.base_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.base_rotation': ModalityDef(source_column='observation.state', start=3, end=7, rotation_type="quaternion"),

            'action.eef_position': ModalityDef(source_column='world_delta_action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='world_delta_action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='world_delta_action', start=6, end=7),
            'action.base_motion': ModalityDef(source_column='action', start=7, end=11, absolute=False),
            'action.control_mode': ModalityDef(source_column='action', start=11, end=12),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={
                    # "state.eef_rotation": TARGET_STATE_ROTATION_TYPE,
                    "state.base_rotation": TARGET_STATE_ROTATION_TYPE
                },
                # normalization_modes=self.action_normalization_modes,
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                # target_rotations={"action.eef_rotation": TARGET_ACTION_ROTATION_TYPE},
                normalization_modes=self.action_normalization_modes,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1SonicBrainco6DofDataConfig(BaseDataConfig):
    """DataConfig for the Unitree G1 + BrainCo Revo2 (6-DOF hand) SONIC-latent
    fine-tune dataset ("sep-10-2026-cleaned-merged", LeRobot v2.1, 535
    episodes / ~182k frames @ 50Hz, teleop'd through NVlabs
    GR00T-WholeBodyControl's GEAR-SONIC whole-body controller).

    Distinct from LiberoNoNormDataConfig/RobocasaHumanDataConfig: those
    predict physical eef/gripper/base commands. This dataset instead
    supervises on SONIC's 64-dim continuous motion-token latent
    (`action.motion_token`, encoded live by the SONIC encoder during
    teleop, script_config.is_using_encoder=true) plus the two 6-DOF BrainCo
    Revo2 hand commands. A separately-trained SONIC decoder
    (model_decoder.onnx, run at 50Hz on the robot, outside this training
    pipeline) turns the latent back into joint commands at deploy time.
    `action.wbc` (41-dim raw joint targets) is a debug/reference field in
    this dataset and is intentionally NOT declared as a modality here, so
    it is never loaded or used as a target.

    Action target = 64 (motion_token) + 6 (left_hand_joints) + 6
    (right_hand_joints) = 76-dim, sampled over `action_chunk_length` steps
    (pass `--action_chunk_length 40` at the CLI -- 40 steps @ 50Hz = 0.8s
    lookahead; BaseDataConfig.get_sampling_indices() applies this to every
    ACTION_KEYS entry uniformly via `action_indices=list(range(chunk_len))`
    in LeRobotIterableDataset.__init__, so there is nothing else to wire up
    for chunking).

    Video: only the two working camera streams are declared
    (`depth_camera_rgb`, `webcam`). `external_view_camera` is intentionally
    omitted from VIDEO_SOURCE_COLUMNS/VIDEO_KEYS -- add_video_modality()
    (BaseDataConfig, above) only ever adds keys from VIDEO_SOURCE_COLUMNS,
    and LeRobotIterableDataset._extract_and_sample_step() only opens/decodes
    the mp4 for a video key if it appears in `modality_defs` (i.e. was
    returned by define_modalities()) -- so the excluded camera's video is
    never opened, decoded, or otherwise touched. This is a config-only
    exclusion; no source patch was needed for it.

    ================================================================
    NOTE ON STATE COLUMN ORDERING -- THIS IS A GUESS, NOT VERIFIED
    ================================================================
    The real dataset's meta/info.json (which would give the authoritative
    `observation.state.names` ordering) was deleted locally and is not
    available to this integration -- per the task schema, observation.state
    is 41-dim = legs(12) + waist(3) + left_arm+hand(13) + right_arm+hand(13).
    The split below (left_leg/right_leg = 6/6, left_arm/hand = 7/6,
    right_arm/hand = 7/6) assumes: (a) legs split evenly left-then-right
    (6 DOF/leg is the typical Unitree G1 leg DOF count: hip roll/pitch/yaw,
    knee, ankle pitch/roll), (b) each arm+hand group of 13 is arm(7,
    typical G1 arm DOF: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/
    yaw) + hand(6, BrainCo Revo2's own DOF count, given in the schema), and
    (c) group order follows the RLDX-1 fork's own group-naming convention
    for this same dataset (left_leg, right_leg, waist, left_arm, left_hand,
    right_arm, right_hand -- see the sibling RLDX-1 adaptation's
    unitree_g1_sonic_brainco6dof_config.py). VERIFY THIS against the real
    dataset's meta/info.json `observation.state.names` before training --
    if the true order differs, only the `start`/`end` slice indices below
    need to change, nothing else in this class.

    ================================================================
    NOTE ON UNIFIED_MAPPING SLOT CHOICES -- SEE docs/unified_action_space.md
    ================================================================
    This embodiment doesn't cleanly fit the documented slot layout (that
    layout is written for single/dual-arm tabletop manipulators with
    optional wheeled base -- LIBERO/RoboCasa -- not a bipedal humanoid with
    legs, nor a latent action target). Choices made here:
      - left_arm (57-70... actually 57-64) / right_arm (50-57): use the
        documented 7-DoF dual-arm "arm_joint_position" slots as-is -- this
        dataset's 7-DoF arms are exactly what that slot was sized for.
      - left_hand (32-38) / right_hand (20-26): use the documented 6-DoF
        "dexhand_position" slots as-is -- BrainCo Revo2 is 6-DoF, an exact
        fit, for both STATE (proprioception) and ACTION (the actual
        76-dim training target's hand components).
      - waist (66-69): documented "waist_position" slot, exact fit (3-dim).
      - left_leg (76-82) / right_leg (82-88): the documented layout has NO
        slot for bipedal leg state at all (its "Mobile Base" slots, 70-76,
        are sized and semantically meant for a wheeled base position/motion
        command, not 12 dims of leg joint angles). Placed both legs inside
        the documented "76-90: Reserved for future embodiments and
        extensions" block, which is both big enough (14 dims for our 12)
        and explicitly meant for exactly this situation. This is a genuine
        extension of the unified space, not an existing convention --
        flagging it plainly rather than presenting it as settled.
      - motion_token (90-154, 64 dims, ACTION space only): SONIC's latent
        has no physical-joint semantics at all, so none of the documented
        named slots apply. Placed inside the documented "90-200: Human
        Hands" block (MANO parameters), which this embodiment never
        populates (we have no MANO/human-hand data) -- i.e. genuinely
        free space for this embodiment, not a collision with anything else
        this recipe uses. Picked the first 64 dims of that block (90-154),
        leaving 154-200 untouched. Like the leg placement above, this is a
        deliberate reuse of otherwise-unused space, not an official slot --
        flagged rather than silently assumed.
      - Only ONE state slot conflict to be aware of if you later add MORE
        embodiments to the same training run: dexhand_position (20-26,
        32-38) is also documented as the *generic* dexterous-hand slot, so
        another 6-DoF-hand embodiment sharing this run would collide here
        unless it also happens to be semantically "the same kind of hand
        slot" -- acceptable for a single-embodiment fine-tune (this recipe),
        would need review before folding into a multi-embodiment YAML.
    """

    VIDEO_KEYS = ['video.depth_camera_rgb', 'video.webcam']
    VIDEO_SOURCE_COLUMNS = {
        'video.depth_camera_rgb': 'observation.images.depth_camera_rgb',
        'video.webcam': 'observation.images.webcam',
        # 'video.external_view_camera' deliberately NOT declared -- see
        # class docstring "Video" section above.
    }

    STATE_KEYS = [
        'state.left_leg', 'state.right_leg', 'state.waist',
        'state.left_arm', 'state.left_hand',
        'state.right_arm', 'state.right_hand',
    ]
    ACTION_KEYS = [
        'action.motion_token', 'action.left_hand_joints', 'action.right_hand_joints',
    ]
    LANGUAGE_KEYS = ['language.instruction']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        # --- state: proprioception, mapped into the 200-dim unified STATE space ---
        'state.right_arm':  (50, 57),   # documented arm_joint_position (7-DoF), "right"
        'state.left_arm':   (57, 64),   # documented left_arm_joint_position (7-DoF)
        'state.waist':      (66, 69),   # documented waist_position (3-dim)
        'state.right_hand': (20, 26),   # documented dexhand_position (6-DoF), "right"
        'state.left_hand':  (32, 38),   # documented left_dexhand_position (6-DoF)
        'state.left_leg':   (76, 82),   # NOT an official slot -- see docstring
        'state.right_leg':  (82, 88),   # NOT an official slot -- see docstring

        # --- action: the 76-dim training target, mapped into the 200-dim unified ACTION space ---
        'action.motion_token':       (90, 154),  # NOT an official slot -- see docstring
        'action.right_hand_joints':  (20, 26),    # same dexhand_position slot as state, action-space instance
        'action.left_hand_joints':   (32, 38),    # same left_dexhand_position slot as state, action-space instance
    }

    # Raw pass-through for every key (no mean/std/min-max/binary transform).
    #
    # motion_token: SONIC's own learned continuous latent space. We have no
    # visibility into its true bounds without the real dataset's
    # meta/stats.json (deleted locally), and RLDX-1's own adaptation of this
    # same dataset made the identical choice (ActionRepresentation.ABSOLUTE,
    # no delta/scale transform) -- mirrored here for consistency.
    #
    # left_hand_joints/right_hand_joints: schema says these are already
    # "normalized_0_to_1" by the teleop pipeline. Deliberately NOT using
    # Being-H05's 'binary' mode here even though RoboCasaHumanDataConfig
    # uses 'binary' for its (single-dim, truly open/close) gripper -- these
    # are 6 continuous per-finger joint values, not a single open/close
    # flag, so thresholding at 0.5 would destroy real grasp-aperture
    # information. Passing the already-[0,1]-scaled values through
    # unnormalized is the correct choice here, not an oversight.
    #
    # right_hand_joints in particular: in the ORIGINAL 535-episode dataset,
    # this column is exactly 0.0 (min==max==0.0) across all 535
    # episodes/182031 frames. Per the task owner, that isn't purely an
    # artifact of the old task set: the right hand's own DOF range is
    # hardware-limited and can't close into a full fist or a functional
    # handshake grip, even in newer episodes (hug/handshake/fist-bump) where
    # the right ARM does move (that motion is carried by the SONIC
    # motion-token / observation.state's right_arm group, not this field).
    # So this field may stay a narrow/near-degenerate signal even in the
    # expanded (~1000-episode) dataset -- don't assume it suddenly spans a
    # full functional range, but also don't hardcode "always exactly 0.0" as
    # fact for the new data: re-profile real min/max/std once the new
    # dataset's stats are available, rather than trusting this note or the
    # old numbers. Still declared as part of the 76-dim action target per
    # this recipe either way (not silently dropped). (Same note as RLDX-1's
    # adaptation of this dataset.)
    state_normalization_modes = {}
    action_normalization_modes = {}

    def get_feature_meta(self):
        return {
            'state.left_leg': ("6-d absolute left leg joint position", 6),
            'state.right_leg': ("6-d absolute right leg joint position", 6),
            'state.waist': ("3-d absolute waist joint position", 3),
            'state.left_arm': ("7-d absolute left arm joint position", 7),
            'state.left_hand': ("6-d absolute left hand (BrainCo Revo2) joint position, normalized_0_to_1", 6),
            'state.right_arm': ("7-d absolute right arm joint position", 7),
            'state.right_hand': ("6-d absolute right hand (BrainCo Revo2) joint position, normalized_0_to_1", 6),
            'action.motion_token': ("64-d continuous SONIC motion-token latent (not a physical joint)", 64),
            'action.left_hand_joints': ("6-d absolute left hand (BrainCo Revo2) joint command, normalized_0_to_1", 6),
            'action.right_hand_joints': ("6-d absolute right hand (BrainCo Revo2) joint command, normalized_0_to_1 -- all-zero in the original 535-ep dataset; likely still narrow/near-degenerate in the expanded dataset due to a hardware DOF limit, not verified fact -- see class docstring", 6),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns.

        Column names below (observation.state, action.motion_token,
        teleop.left_hand_joints, teleop.right_hand_joints) are the literal
        LeRobot v2.1 Parquet column names given in the task schema --
        action.motion_token/teleop.*_hand_joints are each their OWN column
        (unlike LIBERO/RoboCasa's single sliced 'action' column), so each
        gets start=0 covering its full native width.
        """
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            # --- state: sliced out of the single 41-dim observation.state column ---
            # See class docstring "NOTE ON STATE COLUMN ORDERING" -- these
            # start/end offsets are a documented guess, not verified.
            'state.left_leg':   ModalityDef(source_column='observation.state', start=0,  end=6),
            'state.right_leg':  ModalityDef(source_column='observation.state', start=6,  end=12),
            'state.waist':      ModalityDef(source_column='observation.state', start=12, end=15),
            'state.left_arm':   ModalityDef(source_column='observation.state', start=15, end=22),
            'state.left_hand':  ModalityDef(source_column='observation.state', start=22, end=28),
            'state.right_arm':  ModalityDef(source_column='observation.state', start=28, end=35),
            'state.right_hand': ModalityDef(source_column='observation.state', start=35, end=41),

            # --- action: each target lives in its own dataset column ---
            'action.motion_token': ModalityDef(
                source_column='action.motion_token', start=0, end=64, absolute=True),
            'action.left_hand_joints': ModalityDef(
                source_column='teleop.left_hand_joints', start=0, end=6, absolute=True),
            'action.right_hand_joints': ModalityDef(
                source_column='teleop.right_hand_joints', start=0, end=6, absolute=True),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes,
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "libero_nonorm": LiberoNoNormDataConfig,
    "robocasa_human": RobocasaHumanDataConfig,
    "g1_sonic_brainco6dof": UnitreeG1SonicBrainco6DofDataConfig,
}
