#!/usr/bin/env python3
"""Fit a read-only approximation of passive BrainCo joints for shadow inference."""
import argparse
import json
from pathlib import Path

import numpy as np

# In 46D coordinates: arms plus six actuated joints in each hand.
MOTORS = np.array(list(range(7)) + [7, 8, 11, 14, 17, 20]
                  + list(range(23, 30)) + [30, 31, 34, 37, 40, 43])
PASSIVE = np.array([i for i in range(46) if i not in set(MOTORS)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('/data/skurchev/workspace/datasets/hf_release_cosmos3_140_200'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    specs = [('batch_140_sam3d_objects', 'training_data_g1brainco_robohamer.npz'),
             ('batch_200_oreo_grasp', 'qpos_robohamer.npz')]
    episodes = []
    for batch, filename in specs:
        for p in sorted((args.source / batch).iterdir()):
            if p.is_dir():
                with np.load(p / filename, allow_pickle=False) as d:
                    q = np.asarray(d['qpos'][:, 15:61], dtype=np.float64)
                assert q.ndim == 2 and q.shape[1] == 46 and np.isfinite(q).all()
                episodes.append(q)
    assert len(episodes) == 340, len(episodes)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(episodes))
    train = np.concatenate([episodes[i] for i in order[:306]])
    test = np.concatenate([episodes[i] for i in order[306:]])
    # Regress each passive joint from the six measured motors of its own hand.
    coefficients = np.zeros((20, 7), dtype=np.float64)
    rmse = []
    for k, index in enumerate(PASSIVE):
        side = [7, 8, 11, 14, 17, 20] if index < 23 else [30, 31, 34, 37, 40, 43]
        x = np.column_stack((np.ones(len(train)), train[:, side]))
        coefficients[k] = np.linalg.lstsq(x, train[:, index], rcond=None)[0]
        y = np.column_stack((np.ones(len(test)), test[:, side])) @ coefficients[k]
        rmse.append(float(np.sqrt(np.mean((y - test[:, index]) ** 2))))
    result = {'motor_indices': MOTORS.tolist(), 'passive_indices': PASSIVE.tolist(),
              'passive_coefficients': coefficients.tolist(),
              'passive_min': train[:, PASSIVE].min(axis=0).tolist(),
              'passive_max': train[:, PASSIVE].max(axis=0).tolist(),
              'validation_rmse_rad': rmse, 'validation_max_rmse_rad': max(rmse),
              'episodes_train': 306, 'episodes_test': 34}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'WROTE {args.output}; held-out passive RMSE mean={np.mean(rmse):.4f}, max={max(rmse):.4f} rad')


if __name__ == '__main__':
    main()
