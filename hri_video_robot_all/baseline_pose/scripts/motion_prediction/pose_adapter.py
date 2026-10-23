"""COCO WholeBody -> canonical_upper_body_v1; no H36M conversion."""
from dataclasses import dataclass

import numpy as np

JOINT_NAMES = ('pelvis', 'neck_or_shoulder_center', 'left_shoulder',
               'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
SOURCE_MAPPING = ((11, 12), (5, 6), (5,), (6,), (7,), (8,), (9,), (10,))
ROOT_INDEX = 1
Z_SCALE = 2.1744869


@dataclass
class CanonicalSequence:
    pose: np.ndarray
    confidence: np.ndarray
    visibility: np.ndarray
    root_relative_pose: np.ndarray
    root_image_position: np.ndarray
    metadata: dict


def adapt(poses, image_width, image_height, confidence_threshold=0.3, z_scale=Z_SCALE):
    if not all(np.isfinite(x) and x > 0 for x in (image_width, image_height, z_scale)):
        raise ValueError('Image dimensions and z_scale must be finite and positive')
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must be in [0,1]')
    xy = np.asarray(poses['keypoints_2d'], dtype=float)
    xyz = np.asarray(poses['keypoints_3d'], dtype=float)
    scores = np.asarray(poses['scores'], dtype=float)
    t = len(xy)
    if xy.shape != (t, 133, 2) or xyz.shape != (t, 133, 3) or scores.shape != (t, 133):
        raise ValueError('Expected COCO133 arrays [T,133,2/3] and scores [T,133]')
    visible = np.asarray(poses['visibility'], dtype=bool)
    if visible.shape != scores.shape:
        raise ValueError('visibility shape does not match scores')
    valid = visible & np.isfinite(scores) & (scores >= confidence_threshold)
    valid &= np.isfinite(xy).all(-1) & np.isfinite(xyz[..., 2])
    if 'person_detected' in poses:
        detected = np.asarray(poses['person_detected'], dtype=bool)
        if detected.shape != (t,):
            raise ValueError('person_detected must be [T]')
        valid &= detected[:, None]
    raw = np.concatenate((xy / [image_width, image_height], xyz[..., 2:] / z_scale), -1)
    pose = np.stack([raw[:, ids, :].mean(1) for ids in SOURCE_MAPPING], 1)
    confidence = np.stack([scores[:, ids].min(1) for ids in SOURCE_MAPPING], 1)
    visibility = np.stack([valid[:, ids].all(1) for ids in SOURCE_MAPPING], 1)
    # Every relative depth depends on both shoulders, including its confidence.
    visibility &= visibility[:, ROOT_INDEX, None]
    confidence = np.minimum(confidence, confidence[:, ROOT_INDEX, None])
    confidence = np.where(visibility, np.clip(confidence, 0, 1), 0)
    root = pose[:, ROOT_INDEX, :2].copy()
    pose[..., 2] -= pose[:, ROOT_INDEX, 2, None]
    pose[~visibility] = np.nan
    root[~visibility[:, ROOT_INDEX]] = np.nan
    relative = pose.copy()
    relative[..., :2] -= root[:, None, :]
    metadata = {
        'representation': 'canonical_upper_body_v1',
        'coordinate_frame': 'normalized_image_2p5d',
        'joint_names': list(JOINT_NAMES),
        'source_joint_mapping': dict(zip(JOINT_NAMES, map(list, SOURCE_MAPPING))),
        'root_joint': JOINT_NAMES[ROOT_INDEX], 'image_width': image_width,
        'image_height': image_height, 'z_scale': z_scale,
        'normalization': {'x': 'keypoints_2d.x / image_width (right)',
                          'y': 'keypoints_2d.y / image_height (down)',
                          'z': '(keypoints_3d.z - shoulder_midpoint.z) / z_scale',
                          'root_relative': 'pose.xy - root_image_position; z unchanged'},
        'units': 'dimensionless; mixed normalized image/depth axes, not Euclidean physical space',
        'depth_source': 'local RTMPose3d.postprocess: z=(decoded_z/(384/2)-1)*2.1744869',
        'confidence_rule': 'minimum of source joints and both root shoulders; invalid=0',
        'visibility_rule': 'source visibility AND confidence AND finite coordinates AND valid root',
        'limitations': ['RTMW3D pseudo-labels, not mocap ground truth',
                        'Z scale is a model convention, not calibrated robot depth',
                        'No interpolation; missing coordinates are NaN with visibility=false',
                        'Per-frame person selection can switch identities; no tracking',
                        'Pretrained siMLPe input skeleton/units are incompatible',
                        'For VLA conditioning, not direct impedance-controller coordinates'],
    }
    return CanonicalSequence(pose, confidence, visibility, relative, root, metadata)
