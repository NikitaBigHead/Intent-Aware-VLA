#!/usr/bin/env python3
"""One GR00T inference from recorded robot state; no robot command interface."""
import argparse
import json
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
import gr00t.policy.gr00t_policy as policy_module
from gr00t.policy.gr00t_policy import Gr00tPolicy

ARM_LEFT = np.arange(0, 7)
ARM_RIGHT = np.arange(23, 30)
HAND_LEFT = np.array([7, 8, 11, 14, 17, 20])
HAND_RIGHT = np.array([30, 31, 34, 37, 40, 43])
# BrainCo motor order: Thumb, ThumbAux, Index, Middle, Ring, Pinky.
RAD_MAX = np.array([1.5184, 1.0472, 1.4661, 1.4661, 1.4661, 1.4661])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--mapping', type=Path, required=True)
    ap.add_argument('--model-path', default='/data/checkpoints/g1_robohamer_340_train/checkpoint-30000')
    ap.add_argument('--prompt', default='pick up the apple and place it down')
    ap.add_argument('--output', type=Path, default=Path('/data/datasets/robohamer_shadow_output.npz'))
    args = ap.parse_args()

    with np.load(args.input, allow_pickle=False) as d:
        img, body = d['image'], d['body_q']
        left, right = d['left_hand_norm'], d['right_hand_norm']
    assert body.shape == (29,) and left.shape == right.shape == (6,)
    q = np.zeros(46, dtype=np.float64)
    q[ARM_LEFT], q[ARM_RIGHT] = body[15:22], body[22:29]
    q[HAND_LEFT], q[HAND_RIGHT] = left * RAD_MAX, right * RAD_MAX
    mapping = json.loads(args.mapping.read_text())
    passive = np.asarray(mapping['passive_indices'], dtype=int)
    coeff = np.asarray(mapping['passive_coefficients'], dtype=float)
    assert coeff.shape == (20, 7) and len(passive) == 20
    for k, index in enumerate(passive):
        motor = HAND_LEFT if index < 23 else HAND_RIGHT
        q[index] = np.clip(np.dot(coeff[k], np.r_[1.0, q[motor]]),
                           mapping['passive_min'][k], mapping['passive_max'][k])
    assert np.isfinite(q).all()
    # Match the converted video's 640x392 dimensions for this shadow probe.
    img = np.asarray(Image.fromarray(img).resize((640, 392), Image.Resampling.BILINEAR))
    obs = {'video': {'ego_view': img[None, None]},
           'state': {'joint_positions': q.astype(np.float32)[None, None]},
           'language': {'annotation.human.task_description': [[args.prompt]]}}
    tags = [obj for obj in vars(policy_module).values()
            if isinstance(obj, type) and issubclass(obj, Enum) and hasattr(obj, 'NEW_EMBODIMENT')]
    assert len(tags) == 1, f'Could not resolve GR00T EmbodimentTag: {tags}'
    policy = Gr00tPolicy(embodiment_tag=tags[0].NEW_EMBODIMENT,
                        model_path=args.model_path, device='cuda')
    action, _ = policy.get_action(obs)
    key = 'action.joint_positions' if 'action.joint_positions' in action else 'joint_positions'
    output = np.asarray(action[key], dtype=np.float64).reshape(-1, 46)
    assert output.shape[0] == 16 and np.isfinite(output).all(), output.shape
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, observed_q=q, predicted_q=output, image=img)
    first = output[0]
    print('SHADOW ONLY: no commands sent to robot')
    print('output:', args.output, 'prediction:', output.shape)
    print('observed left/right arms:', np.round(q[ARM_LEFT], 3), np.round(q[ARM_RIGHT], 3))
    print('predicted left/right arms:', np.round(first[ARM_LEFT], 3), np.round(first[ARM_RIGHT], 3))
    print('predicted left/right hand motors [rad]:', np.round(first[HAND_LEFT], 3), np.round(first[HAND_RIGHT], 3))
    print('max arm change over first action [rad]:',
          round(float(max(np.max(np.abs(first[ARM_LEFT]-q[ARM_LEFT])),
                          np.max(np.abs(first[ARM_RIGHT]-q[ARM_RIGHT])))), 4))


if __name__ == '__main__':
    main()
