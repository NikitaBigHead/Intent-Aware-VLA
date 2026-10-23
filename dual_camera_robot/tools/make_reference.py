"""Build reference/ from the ORIGINAL pipeline (run inside the lab repository, not on the robot).

Uses artifacts/dual_camera_attention_v1/{dataset.npz, model.py, dual_camera_attention_best.pt} exactly as
train_eval.py does, and copies the two camera videos and the saved RTMW3D poses of each episode.
    python -B dual_camera_robot/tools/make_reference.py 105 355 72
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
A = REPO / 'artifacts/dual_camera_attention_v1'
F = REPO / 'artifacts/full_first_iter_finetuned_v1'
D = REPO / 'datasets/first_iter_dataset/sep-10-2026-cleaned-merged'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(A))
from model import MultiCam  # noqa: E402  original model code
from scripts.motion_prediction.maskaware.decision import DEFAULT, decide  # noqa: E402
from scripts.motion_prediction.sequence_utils import sha256  # noqa: E402

CL = ['hug', 'handshake', 'fist_bump', 'no_interaction']


def reliability(x):  # train_eval.py
    j = x[..., :120].reshape(*x.shape[:-1], 8, 15)[..., 11]; hobs = x[..., [126, 134]]
    return np.concatenate((j, hobs), -1).mean(-1) * (x[..., 140] > 0)


def main(eps):
    d = dict(np.load(A / 'dataset.npz'))
    ck = torch.load(A / 'dual_camera_attention_best.pt', map_location='cpu', weights_only=False)
    model = MultiCam([141, 141]); model.load_state_dict(ck['action_state_dict']); model.eval()
    audit = {s['episode_index']: s for s in json.loads((F / 'audit.json').read_text())['sources']}
    records = {json.loads(l)['episode_index']: json.loads(l) for l in (D / 'meta/interaction_metadata.jsonl').read_text().splitlines() if l.strip()}
    index = []
    for e in eps:
        i = np.flatnonzero(d['episode_index'] == e)
        xs = [torch.tensor((d[k][i].astype(np.float32) - n['mean']) / n['std']) for k, n in zip(('x_webcam', 'x_hand'), ck['norm'])]
        rel = torch.tensor(np.stack([reliability(d['x_webcam'][i]), reliability(d['x_hand'][i])], 1), dtype=torch.float32)
        with torch.no_grad():
            logits, _, w = model(xs, rel, torch.zeros(len(i), 8, 3))
        p = torch.softmax(logits, -1).numpy()
        labels, reasons, adjusted = decide(p, d['action_quality_union'][i], d['episode_index'][i], d['origin_s'][i], ck['threshold'], CL, DEFAULT)
        out = PKG / f'reference/episode_{e:06d}'; out.mkdir(parents=True, exist_ok=True)
        for cam, key in (('person', 'observation.images.webcam'), ('hand', 'observation.images.depth_camera_rgb')):
            shutil.copy2(D / f'videos/chunk-000/{key}/episode_{e:06d}.mp4', out / f'{cam}.mp4')
        shutil.copy2(audit[e]['raw_pose'], out / 'person_raw_pose.npz')
        shutil.copy2(A / f'poses_hand_camera/episode_{e:06d}.npz', out / 'hand_raw_pose.npz')
        np.savez_compressed(out / 'reference.npz', origin_s=d['origin_s'][i], x_webcam=d['x_webcam'][i], x_hand=d['x_hand'][i],
                            action_quality_union=d['action_quality_union'][i], probabilities=p, labels=labels, reasons=reasons,
                            adjusted=adjusted, hand_attention_last=w[:, 1, -1].numpy(), target=d['interaction_target'][i])
        acc = labels[labels >= 0]
        info = dict(episode_index=e, person=records[e]['person'], action=records[e]['interaction_class'], windows=int(len(i)),
                    original_majority=CL[np.bincount(acc, minlength=4).argmax()] if len(acc) else 'unknown',
                    original_accepted=int(len(acc)), threshold=float(ck['threshold']),
                    source_checkpoint_sha256=sha256(A / 'dual_camera_attention_best.pt'), source_dataset_sha256=sha256(A / 'dataset.npz'))
        (out / 'info.json').write_text(json.dumps(info, indent=1, ensure_ascii=False))
        index.append(info)
        print(info)
    (PKG / 'reference/index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [105, 355, 72])
