"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'

# lisa method
ACTION_TOKEN_IDX = 32001

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

FRACTAL_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 16,
    "PROPRIO_DIM": 16,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

G1_EE_6D_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_STACK_BLOCK_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Unitree G1 + BrainCo Revo2 (6-DOF hands), SONIC motion-token dataset
# "sep-10-2026-cleaned-merged" (535 episodes, ~182k frames, 50 Hz).
#
#   NUM_ACTIONS_CHUNK = 40   -- 40 steps @ 50 Hz = 0.8s lookahead. This directly drives BOTH
#     sides of the pipeline that must agree: the RLDS dataloader's
#     `future_action_window_size=NUM_ACTIONS_CHUNK-1` (datasets.py, RLDSDataset.__init__) and
#     the DiT action head's `self.action_horizon = NUM_ACTIONS_CHUNK`
#     (DiT_ActionHeader.py) -- both import this constant directly, so nothing else needs to
#     change to pick up a new chunk size. (unifolm_vla_train.yaml also has a
#     framework.action_model.action_horizon field, but it is unused/stale -- the model reads
#     the chunk size from this file, not the yaml. Left the yaml alone since editing it would
#     be misleading busywork.)
#   ACTION_DIM = 76          -- 64 (action.motion_token, SONIC latent) + 6
#     (teleop.left_hand_joints) + 6 (teleop.right_hand_joints). `action.wbc` (41-dim raw
#     joint targets) is intentionally NOT included -- debug/reference field only.
#     In the original 535-episode dataset, teleop.right_hand_joints is exactly 0.0
#     (min==max==0.0) across every episode. CORRECTED (task owner, after the dataset
#     grew to ~1000 episodes): that isn't purely an artifact of the old task set. The
#     expanded dataset adds hug/handshake/fist-bump tasks where the right ARM does
#     move -- but that motion is carried by the 64-dim action.motion_token channel /
#     observation.state's right-arm proprio dims, NOT by right_hand_joints. The right
#     hand's own DOF range is hardware-limited and reportedly can't close into a full
#     fist or a functional handshake grip, even in those newer episodes. So this field
#     may STILL be a narrow/near-degenerate signal in the expanded dataset -- don't
#     assume it suddenly spans a full functional range, but also don't hardcode
#     "always exactly 0.0" as fact for the new data: re-profile real per-dim
#     min/max/std once the new dataset's stats are available (get_dataset_statistics
#     in rlds/utils/data_utils.py computes these automatically on first load). This
#     also changes the BOUNDS_Q99 risk profile below -- see the note added there.
#     Still included regardless of its actual actuation range -- dropping it would
#     desync ACTION_DIM=76 (see ActionEncoding.SONIC_LATENT_G1_BRAINCO6DOF in
#     oxe/configs.py for how normalization handles it).
#   PROPRIO_DIM = 41         -- legs(12) + waist(3) + left arm+hand(13) + right arm+hand(13),
#     read straight out of `observation.state` (no reconstruction/splitting needed).
#   ACTION_PROPRIO_NORMALIZATION_TYPE = BOUNDS_Q99  -- ** real, flagged risk, read before
#     changing **. The two existing G1 entries above disagree with each other (G1_CONSTANTS
#     uses BOUNDS = plain min/max scaling to [-1, 1]; G1_STACK_BLOCK_CONSTANTS uses
#     BOUNDS_Q99 = 1st/99th-percentile scaling, clipped to [-1, 1] beyond that). Both of
#     those datasets' actions are raw or R6-encoded joint/EEF values, where a hard min/max
#     is physically meaningful (a joint really does have a mechanical limit). `motion_token`
#     is NOT that -- it's SONIC's continuous latent embedding, live-encoded during teleop,
#     with no known physical bound. NVlabs GR00T-WholeBodyControl's own inference code
#     sanity-checks `|motion_token| > 1.25` as roughly the expected operating range, which
#     suggests the distribution is roughly bounded in practice but is NOT a hard clip -- a
#     single noisy/outlier encoder frame anywhere in the 535 episodes could have a much
#     larger magnitude. Plain BOUNDS (min/max) normalization is not robust to that: one
#     outlier frame stretches the [min, max] range used for every other frame, crushing the
#     other 533 episodes' worth of the signal toward 0 post-normalization. BOUNDS_Q99 is
#     robust to exactly this failure mode (it clips the outer 1% instead of anchoring on the
#     single most extreme sample), which is why it's chosen here over BOUNDS -- but this has
#     NOT been verified against the real dataset (it was deleted locally before this
#     adaptation; get_dataset_statistics in rlds/utils/data_utils.py computes these
#     percentiles from the actual data the first time this dataset is loaded, so the real
#     q01/q99 values are unknown until then). If the real motion_token distribution turns
#     out to have no meaningful outliers, BOUNDS_Q99 and BOUNDS would behave almost
#     identically anyway, so this choice is low-downside; the failure mode it guards against
#     (BOUNDS silently crushing the useful range) is the one that would be hard to notice
#     without directly inspecting normalized values.
#   ADDED NOTE re: right_hand_joints under BOUNDS_Q99 (see the ACTION_DIM correction
#     above) -- this normalization type is applied uniformly across all 76 action dims,
#     including right_hand_joints. data_utils.py's zeros_mask only collapses a dim to a
#     constant 0 when min==max EXACTLY. In the original 535-episode dataset that held
#     (exactly 0.0 everywhere), so right_hand_joints safely became a no-op post-
#     normalization. In the expanded dataset, if right_hand_joints turns out to be
#     narrow-but-not-exactly-constant (e.g. small hardware jitter around a resting
#     value, rather than a bit-exact 0.0), min==max will be False, zeros_mask won't
#     trigger, and BOUNDS_Q99 will instead normalize using that dim's real (possibly
#     tiny) q01/q99 range -- dividing by a near-zero range amplifies small sensor noise
#     into large post-normalization swings. Not fixable in advance without the real
#     per-dim stats; flagging so this is checked (e.g. print per-dim q01/q99 for the 6
#     right_hand_joints dims) once the new dataset's statistics are computed, rather
#     than assumed safe by analogy with the old data.
G1_SONIC_BRAINCO6DOF_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 40,
    "ACTION_DIM": 76,
    "PROPRIO_DIM": 41,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()
    print(cmd_args)
    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "fractal" in cmd_args:
        return "FRACTAL"
    elif "ee_6d" in cmd_args:
        return "G1_EE_6D"
    elif "sonic" in cmd_args:
        # Matches on the data_mix name "g1_sonic_brainco6dof" passed on the CLI (e.g.
        # `--datasets.vla_data.data_mix g1_sonic_brainco6dof`). "sonic" doesn't collide with
        # any other branch's keyword here, so this is safe regardless of ordering -- placed
        # before the "joint"/"stack_block" checks only for readability, not correctness.
        return "G1_SONIC_BRAINCO6DOF"
    elif "joint" in cmd_args:
        return "G1"
    elif "stack_block" in cmd_args:
        return "G1_STACK_BLOCK"
    else:
        return "G1_EE_6D"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "FRACTAL":
    constants = FRACTAL_CONSTANTS
elif ROBOT_PLATFORM == "G1_EE_6D":
    constants = G1_EE_6D_CONSTANTS
elif ROBOT_PLATFORM == "G1":
    constants = G1_CONSTANTS
elif ROBOT_PLATFORM == "G1_STACK_BLOCK":
    constants = G1_STACK_BLOCK_CONSTANTS
elif ROBOT_PLATFORM == "G1_SONIC_BRAINCO6DOF":
    constants = G1_SONIC_BRAINCO6DOF_CONSTANTS


# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f" in constants.py NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `training/vla/constants.py`!")
