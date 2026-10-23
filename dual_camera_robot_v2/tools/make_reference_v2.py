"""Build reference/ for the v2 model from the ORIGINAL v2 pipeline (run inside the lab repository, not on the robot).

Source: artifacts/dual_camera_attention_v2_mac/{test_dataset.npz, poses/test, audit.json, final/...model.pt} and the
unchanged model code artifacts/dual_camera_attention_v1/model.py, exactly as train_eval.py scores test-dataset.
    python -B dual_camera_robot_v2/tools/make_reference_v2.py
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
V2 = REPO / 'artifacts/dual_camera_attention_v2_mac'
CKPT = V2 / 'final/dual_webcam_hand_weighted_seed42/model.pt'
sys.path.insert(0, str(REPO))
from scripts.motion_prediction.maskaware.decision import DEFAULT, decide  # noqa: E402
from scripts.motion_prediction.sequence_utils import sha256  # noqa: E402

spec = importlib.util.spec_from_file_location('v1_model', REPO / 'artifacts/dual_camera_attention_v1/model.py')
v1m = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1m)
CL = ['hug', 'handshake', 'fist_bump', 'no_interaction']
CAMS = ('webcam', 'depth_camera_rgb')


def reliability(x):  # train_eval.py
    j = x[..., :120].reshape(*x.shape[:-1], 8, 15)[..., 11]; hobs = x[..., [126, 134]]
    return np.concatenate((j, hobs), -1).mean(-1) * (x[..., 140] > 0)


def pick(per_episode):
    """One correct episode per class from different people + one typical error (handshake -> fist_bump)."""
    chosen, used = [], set()
    for c in ('hug', 'handshake', 'fist_bump'):
        e = next(e for e, v in sorted(per_episode.items(), key=lambda kv: -kv[1]['accepted'])
                 if v['true'] == c and v['primary'] == c and v['person'] not in used)
        chosen.append(int(e)); used.add(per_episode[e]['person'])
    e = next(e for e, v in sorted(per_episode.items(), key=lambda kv: -kv[1]['accepted'])
             if v['true'] == 'handshake' and v['primary'] == 'fist_bump' and v['person'] not in used)
    return chosen + [int(e)]


def main():
    td = dict(np.load(V2 / 'test_dataset.npz'))
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = v1m.MultiCam([141, 141]); model.load_state_dict(ck['action_state_dict']); model.eval()
    audit = {r['episode_index']: r for r in json.loads((V2 / 'audit.json').read_text())['test_dataset']['rows']}
    per_episode = json.loads((V2 / 'test_dataset_per_episode.json').read_text())['final_dual_webcam_hand']
    index = []
    for e in pick(per_episode):
        i = np.flatnonzero(td['episode_index'] == e)
        xs = [torch.tensor((td[k][i].astype(np.float32) - n['mean']) / n['std']) for k, n in zip(('x_webcam', 'x_hand'), ck['norm'])]
        rel = torch.tensor(np.stack([reliability(td['x_webcam'][i]), reliability(td['x_hand'][i])], 1), dtype=torch.float32)
        with torch.no_grad():
            logits, _, w = model(xs, rel, torch.zeros(len(i), 8, 3))
        p = torch.softmax(logits, -1).numpy()
        labels, reasons, adjusted = decide(p, td['action_quality_union'][i], td['episode_index'][i], td['origin_s'][i], ck['threshold'], CL, DEFAULT)
        out = PKG / f'reference/episode_{e:06d}'; out.mkdir(parents=True, exist_ok=True)
        for cam, key in (('person', 'webcam'), ('hand', 'depth_camera_rgb')):
            shutil.copy2(audit[e]['videos'][key], out / f'{cam}.mp4')
        z = np.load(V2 / f'poses/test/episode_{e:06d}.npz')
        for cam, key in (('person', 'webcam'), ('hand', 'depth_camera_rgb')):
            np.savez_compressed(out / f'{cam}_raw_pose.npz', **{k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(key + '__')})
        np.savez_compressed(out / 'reference.npz', origin_s=td['origin_s'][i], x_webcam=td['x_webcam'][i], x_hand=td['x_hand'][i],
                            action_quality_union=td['action_quality_union'][i], probabilities=p, labels=labels, reasons=reasons,
                            adjusted=adjusted, hand_attention_last=w[:, 1, -1].numpy())
        acc = labels[labels >= 0]
        info = dict(episode_index=e, dataset='datasets/test-dataset', person=audit[e]['person'], action=audit[e]['interaction_class'],
                    windows=int(len(i)), original_majority=CL[np.bincount(acc, minlength=4).argmax()] if len(acc) else 'unknown',
                    original_episode_decision=per_episode[str(e)]['primary'], original_accepted=int(len(acc)), threshold=float(ck['threshold']),
                    source_checkpoint_sha256=sha256(CKPT), source_dataset_sha256=sha256(V2 / 'test_dataset.npz'))
        (out / 'info.json').write_text(json.dumps(info, indent=1, ensure_ascii=False))
        index.append(info); print(info)
    (PKG / 'reference/index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
