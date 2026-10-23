"""Dual-camera action recognition (hug / handshake / fist_bump / no_interaction / unknown). Read-only: never commands the robot.

  python -B run_dualcam.py verify                                   # recorded episodes vs original pipeline
  python -B run_dualcam.py replay --person P.mp4 --hand H.mp4       # any recorded episode (both cameras)
  python -B run_dualcam.py robot --host 192.168.123.164 --show      # robot camera server (ZMQ, keys head / ego_view)
  python -B run_dualcam.py cameras --person 0 --hand 1 --show       # two OpenCV sources (USB index / URL)
"""
import argparse
import collections
import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dualcam.engine import CLASSES, DualCamEngine, raw_sample  # noqa: E402
from dualcam.features import TRAIN_SIZE  # noqa: E402
from dualcam.hold import HugUnknownHold  # noqa: E402
from dualcam.intent import IntentPublisher  # noqa: E402
from dualcam.sources import CvPairSource, RobotZmqSource, VideoPairReplay  # noqa: E402

CHECKPOINT = ROOT / 'checkpoint/dual_camera_attention_best.pt'
DET = ROOT / 'models/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'
POSE = ROOT / 'models/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx'
POSE_THRESHOLD = .3
POSE_DT = .1          # 10 Hz pose sampling (every 5th frame of the 50 Hz training recordings)
DECISION_DT = .2      # one decision every 0.2 s
COLORS = {'hug': (60, 180, 255), 'handshake': (80, 220, 80), 'fist_bump': (255, 140, 60), 'no_interaction': (200, 200, 200)}
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]


# ----------------------------------------------------------------------------------------------- setup
def preload_torch_cudnn():
    """A system cuDNN in /lib can shadow torch's copy; ORT then silently falls back to CPU (~150 ms/frame)."""
    try:
        import nvidia.cudnn
    except ImportError:
        return
    lib = os.path.join(nvidia.cudnn.__path__[0], 'lib')
    for name in ('graph', 'engines_precompiled', 'engines_runtime_compiled', 'heuristic', 'ops', 'cnn', 'adv', None):
        p = os.path.join(lib, f'libcudnn_{name}.so.9' if name else 'libcudnn.so.9')
        if os.path.exists(p):
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)


def auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError('confidence must be in [0, 1]')
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('stable frames must be at least 1')
    return value


class Pose:
    """RTMW3D exactly as in training: same ONNX files, threshold 0.3, primary-person rule."""

    def __init__(self, device):
        if device == 'cuda':
            preload_torch_cudnn()
        from scripts.pose3d.infer_video import _select_primary_person
        from scripts.pose3d.rtmw3d_runner import RTMW3DRunner
        self.select = _select_primary_person
        self.runner = RTMW3DRunner(device=device, det_model=DET, pose_model=POSE)
        self.runner.infer(np.zeros((480, 640, 3), np.uint8))  # warm-up outside latency
        self.device = device

    def __call__(self, frame_bgr):
        return self.select(self.runner.infer(frame_bgr), POSE_THRESHOLD)


def empty_person(pose):
    """A 'nobody detected' result; used when a camera frame is missing or out of sync."""
    return pose.select(type('R', (), dict(keypoints_3d=np.zeros((0, 133, 3)), keypoints_2d=np.zeros((0, 133, 2)),
                                           scores=np.zeros((0, 133)), num_people=0))(), POSE_THRESHOLD)


# ----------------------------------------------------------------------------------------------- overlay
def draw_tile(frame, person, title, height=540):
    img = frame.copy()
    if person is not None and person.person_detected:
        kp, vis = person.keypoints_2d, person.visibility
        for a, b in BONES:
            if vis[a] and vis[b]:
                cv2.line(img, tuple(map(int, kp[a])), tuple(map(int, kp[b])), (0, 255, 0), 3)
        for j in list(range(5, 13)) + list(range(91, 133)):
            if vis[j]:
                cv2.circle(img, tuple(map(int, kp[j])), 6 if j < 13 else 3, (0, 255, 0) if j < 13 else (0, 200, 255), -1)
    img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
    cv2.putText(img, title, (10, height - 12), 0, .7, (255, 255, 255), 2)
    return img


def draw(pair, persons, state, status):
    tiles = [draw_tile(pair['person'], persons[0], 'PERSON camera (head)'), draw_tile(pair['hand'], persons[1], 'HANDS camera (ego_view, down)')]
    img = np.hstack(tiles)
    cv2.rectangle(img, (0, 0), (620, 250), (0, 0, 0), -1)
    label = state.get('label', 'warming up (1 s history)')
    cv2.putText(img, label.upper(), (12, 40), 0, 1.2, COLORS.get(label, (0, 0, 255)), 3)
    if state.get('reason'):
        prefix = 'unknown' if state.get('label') == 'unknown' else 'note'
        cv2.putText(img, f"{prefix}: {state['reason']}", (12, 68), 0, .6, (0, 0, 255), 2)
    for k, c in enumerate(CLASSES):
        p = state.get('probabilities', {}).get(c, 0.)
        y = 95 + 24 * k
        cv2.rectangle(img, (190, y - 14), (190 + int(260 * p), y + 2), COLORS[c], -1)
        cv2.putText(img, f'{c} {p:.2f}', (12, y), 0, .55, (255, 255, 255), 1)
    q = state.get('quality', {})
    lines = [f"quality: person {q.get('person_observed', 0):.2f}  hands {q.get('hand_observed', 0):.2f}  union {q.get('union_observed', 0):.2f}"
             f"  attn(hands) {state.get('hand_attention_last', 0):.2f}",
             f"pose {status['pose_hz']:.1f} Hz | skew hand-person {status['skew_ms']:+.0f} ms | processing {status['processing_ms']:.0f} ms"
             f" | capture->decision {status['e2e_ms']}"]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (12, 205 + 22 * i), 0, .48, (255, 255, 255), 1)
    if status.get('desync'):
        cv2.putText(img, 'HANDS CAMERA OUT OF SYNC - frame ignored', (12, img.shape[0] - 45), 0, .8, (0, 0, 255), 2)
    return img


# ----------------------------------------------------------------------------------------------- verify
def verify(args):
    import scripts.pose3d.infer_video  # noqa: F401  (import check)
    device = args.device or auto_device()
    engine_device = 'cpu'  # the GRU classifier runs on CPU (<1 ms), bit-comparable to the reference
    pose = Pose(device) if not args.skip_video else None
    report = []
    for ep_dir in sorted((ROOT / 'reference').glob('episode_*')):
        ref = dict(np.load(ep_dir / 'reference.npz'))
        info = json.loads((ep_dir / 'info.json').read_text())
        row = dict(episode=info['episode_index'], action=info['action'], windows=len(ref['origin_s']),
                   original_majority=info['original_majority'])

        def compare(results, tag):
            byo = {round(r['origin_s'], 4): r for r in results}
            hit = [byo.get(round(o, 4)) for o in ref['origin_s']]
            matched = [(i, r) for i, r in enumerate(hit) if r is not None]
            idx = np.array([i for i, _ in matched])
            out = dict(matched_windows=len(matched))
            if not matched:
                return out
            out['max_abs_diff_x_person'] = float(max(np.abs(r['_x_webcam'] - ref['x_webcam'][i]).max() for i, r in matched))
            out['max_abs_diff_x_hand'] = float(max(np.abs(r['_x_hand'] - ref['x_hand'][i]).max() for i, r in matched))
            out['max_abs_diff_probability'] = float(max(np.abs(r['_probabilities'] - ref['probabilities'][i]).max() for i, r in matched))
            lab = np.array([r['label_index'] for _, r in matched])
            out['same_label_windows'] = int((lab == ref['labels'][idx]).sum())
            out['same_reason_windows'] = int(sum(r['reason'] == ref['reasons'][i] for i, r in matched))
            acc = lab[lab >= 0]
            out['majority'] = CLASSES[np.bincount(acc, minlength=4).argmax()] if len(acc) else 'unknown'
            row[tag] = out
            return out

        # A: saved RTMW3D poses of the recording -> package preprocessing/model/decision (checks the code path exactly)
        rp, rh = dict(np.load(ep_dir / 'person_raw_pose.npz')), dict(np.load(ep_dir / 'hand_raw_pose.npz'))
        eng = DualCamEngine(CHECKPOINT, engine_device)
        results = run_stream(eng, ((float(rp['video_time_s'][i]), {k: rp[k][i] for k in rp}, {k: rh[k][i] for k in rh},
                                    TRAIN_SIZE['person'], TRAIN_SIZE['hand']) for i in range(len(rp['video_time_s']))))
        compare(results, 'saved_poses')

        # B: the two videos -> RTMW3D now -> everything else (full pipeline, what runs on the robot)
        if pose is not None:
            eng = DualCamEngine(CHECKPOINT, engine_device)
            lat = []

            def gen():
                for s in VideoPairReplay(ep_dir / 'person.mp4', ep_dir / 'hand.mp4'):
                    st = time.perf_counter(); a = pose(s['person']); b = pose(s['hand']); lat.append((time.perf_counter() - st) * 1000)
                    yield (s['t_person'], raw_sample(a, s['t_person'], s['frame_index']), raw_sample(b, s['t_hand'], s['frame_index']),
                           s['person'].shape[1::-1], s['hand'].shape[1::-1])
            compare(run_stream(eng, gen()), 'videos_rtmw3d_now')
            row['videos_rtmw3d_now']['pose_ms_both_cameras_median'] = float(np.median(lat))
        report.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (ROOT / 'verify_report.json').write_text(json.dumps(dict(device=device, report=report), indent=1, ensure_ascii=False))
    ok = all(r['saved_poses']['matched_windows'] == r['windows'] and r['saved_poses']['same_label_windows'] == r['windows']
             and r['saved_poses']['same_reason_windows'] == r['windows'] and r['saved_poses']['max_abs_diff_probability'] < 1e-5
             and r['saved_poses']['max_abs_diff_x_person'] < 1e-6 and r['saved_poses']['max_abs_diff_x_hand'] < 1e-6 for r in report)
    video = [r['videos_rtmw3d_now'] for r in report if 'videos_rtmw3d_now' in r]
    agree = sum(v['same_label_windows'] for v in video); total = sum(v['matched_windows'] for v in video)
    print('VERIFY', 'PASS' if ok else 'FAIL', '| saved poses: identical features, labels, unknown reasons; probability diff < 1e-5')
    if video:
        print(f'full pipeline from videos (RTMW3D recomputed): {agree}/{total} windows same label; '
              f"episode majority same: {sum(v['majority'] == r['original_majority'] for v, r in zip(video, report))}/{len(video)}")
    return 0 if ok else 1


def run_stream(engine, samples):
    """Same scheduling as live: add every pair, decide when >= 1 s of history and 0.2 s since the last decision."""
    out = []; first = None; last_decision = -np.inf
    for t, rp, rh, sp, sh in samples:
        if not engine.add(t, rp, rh, tuple(sp), tuple(sh)):
            continue
        first = t if first is None else first
        if t - first >= 1 - 1e-6 and t - last_decision >= DECISION_DT - .05:
            r = engine.step(t)
            if r is not None:
                out.append(r); last_decision = t
    return out


# ----------------------------------------------------------------------------------------------- live / replay
def live(args):
    device = args.device or auto_device()
    pose = Pose(device)
    class_thresholds = {
        label: value for label in CLASSES
        if (value := getattr(args, f'{label}_min_confidence')) is not None
    }
    class_stable_frames = {
        label: value for label in CLASSES
        if (value := getattr(args, f'{label}_stable_frames')) is not None
    }
    engine = DualCamEngine(CHECKPOINT, 'cpu', threshold=args.threshold, class_thresholds=class_thresholds,
                           class_stable_frames=class_stable_frames)
    if args.mode == 'robot':
        src = RobotZmqSource(args.host, args.port, args.person_key, args.hand_key)
    elif args.mode == 'cameras':
        src = CvPairSource(args.person, args.hand)
    else:
        src = None
    replay = iter(VideoPairReplay(args.person, args.hand)) if args.mode == 'replay' else None
    log = open(args.log, 'w') if args.log else None
    intent_pub = IntentPublisher(args.intent_bind) if args.intent_bind else None
    hug_hold = HugUnknownHold(args.hold_hug_on_unknown, args.hug_unknown_hold_seconds)

    def write(rec):
        if log:
            log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()

    write(dict(type='session', wall_time=time.time(), mode=args.mode, source=src.address if src else f'{args.person} | {args.hand}',
               checkpoint=str(CHECKPOINT), threshold=engine.threshold, class_thresholds=engine.class_thresholds,
               class_stable_frames=engine.class_stable_frames,
               pose_device=device, classes=CLASSES,
               training_frame_size=TRAIN_SIZE, cameras=engine.cameras, max_skew_s=args.max_skew,
               hold_hug_on_unknown=args.hold_hug_on_unknown,
               hug_unknown_hold_seconds=args.hug_unknown_hold_seconds))
    thresholds_text = ', '.join(f'{label}={engine.class_thresholds[label]:.2f}' for label in CLASSES)
    print(f'pose on {device}; source {getattr(src, "address", args.mode)}; min confidence: {thresholds_text}', flush=True)
    stability_text = ', '.join(f'{label}={engine.class_stable_frames[label]}' for label in CLASSES)
    print(f'stable decisions required: {stability_text}', flush=True)
    if intent_pub:
        print(f'human intent PUB bound to {intent_pub.endpoint}', flush=True)
    if args.hold_hug_on_unknown:
        print(f'hug-on-unknown hold enabled for {args.hug_unknown_hold_seconds:.1f} s', flush=True)

    writer = None; state = {}; persons = (None, None)
    first = None; last_t = None; last_decision = -np.inf; last_id = None
    pose_times = collections.deque(maxlen=20); proc = collections.deque(maxlen=50); offsets = []
    stats = collections.Counter(); size_warned = False; t_start = time.monotonic()
    next_tick = time.monotonic()
    while True:
        if replay is not None:
            pair = next(replay, None)
            if pair is None:
                break
            pair['received_wall'] = None; pair['has_timestamps'] = True
        else:
            time.sleep(max(0., next_tick - time.monotonic())); next_tick += POSE_DT
            if time.monotonic() > next_tick + POSE_DT:
                next_tick = time.monotonic()
            if args.duration and time.monotonic() - t_start > args.duration:
                break
            pair = src.get()
            if pair is None:
                stats['waiting'] += 1
                if stats['waiting'] % 50 == 1:
                    print(f'waiting for frames from {src.address} (keys seen: {sorted(getattr(src, "keys_seen", []))})', flush=True)
                continue
            if pair['frame_index'] == last_id:
                stats['stale_ticks'] += 1; continue
        last_id = pair['frame_index']
        t = pair['t_person']
        skew = pair['t_hand'] - pair['t_person']
        if last_t is not None and t <= last_t:
            stats['non_monotonic'] += 1; continue
        if last_t is not None and t - last_t > 2.:
            engine.reset(); hug_hold.reset(); first = None; stats['resets_long_gap'] += 1
        sizes = (pair['person'].shape[1::-1], pair['hand'].shape[1::-1])
        if not size_warned:
            size_warned = True
            for cam, s in zip(('person', 'hand'), sizes):
                tr = TRAIN_SIZE[cam]
                if abs(s[0] / s[1] - tr[0] / tr[1]) > .01:
                    print(f'WARNING {cam} camera {s[0]}x{s[1]} has a different aspect than training {tr[0]}x{tr[1]}; '
                          'normalised coordinates will not match training', flush=True)
                elif tuple(s) != tuple(tr):
                    print(f'note: {cam} camera {s[0]}x{s[1]} (training {tr[0]}x{tr[1]}, same aspect)', flush=True)
            write(dict(type='frame_sizes', person=list(sizes[0]), hand=list(sizes[1])))
        st = time.perf_counter()
        pa = pose(pair['person'])
        desync = abs(skew) > args.max_skew
        pb = empty_person(pose) if desync else pose(pair['hand'])
        stats['hand_dropped_desync'] += int(desync)
        engine.add(t, raw_sample(pa, t, last_id), raw_sample(pb, t, last_id), sizes[0], sizes[1])
        persons = (pa, pb); pose_times.append(time.monotonic()); last_t = t
        first = t if first is None else first
        if t - first >= 1 - 1e-6 and t - last_decision >= DECISION_DT - .05:
            r = engine.step(t)
            if r is not None:
                r = hug_hold.apply(r, t)
                last_decision = t
                processing = (time.perf_counter() - st) * 1000 if replay is not None else (time.monotonic() - pair['received']) * 1000
                proc.append(processing)
                c2r = None
                if pair.get('received_wall') is not None:
                    c2r = (pair['received_wall'] - t) * 1000; offsets.append(c2r)
                state = {k: v for k, v in r.items() if not k.startswith('_')}
                write(dict(type='decision', wall_time=time.time(), t_person=t, t_hand=pair['t_hand'], skew_ms=skew * 1000,
                           hand_dropped_desync=desync, capture_to_receive_ms=c2r, processing_ms=processing,
                           capture_to_decision_ms=None if c2r is None else c2r + processing, **state))
                if intent_pub:
                    intent_pub.publish(state, t_person=t, t_hand=pair['t_hand'], skew_ms=skew * 1000)
                if args.print:
                    print(f"{t:10.2f}  {state['label']:<15} {state['reason']:<28} conf {state['confidence']:.2f}  "
                          f"q {state['quality']['union_observed']:.2f}  skew {skew*1000:+.0f} ms  proc {processing:.0f} ms", flush=True)
        if args.show or args.save:
            hz = (len(pose_times) - 1) / (pose_times[-1] - pose_times[0]) if len(pose_times) > 2 else 0.
            e2e = f'{offsets[-1] + proc[-1]:.0f} ms' if offsets and proc else 'n/a'
            img = draw(pair, persons, state, dict(pose_hz=hz, skew_ms=skew * 1000, processing_ms=proc[-1] if proc else 0, e2e_ms=e2e, desync=desync))
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), 10, img.shape[1::-1])
                writer.write(img)
            if args.show:
                cv2.imshow('dual camera action', img)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    summary = dict(type='summary', wall_time=time.time(), **stats,
                   processing_ms_p50=float(np.percentile(proc, 50)) if proc else None,
                   processing_ms_p95=float(np.percentile(proc, 95)) if proc else None,
                   capture_to_receive_ms_min=float(min(offsets)) if offsets else None,
                   note='capture_to_receive uses the robot clock vs this machine clock; valid only if both are NTP/chrony-synced')
    write(summary); print(json.dumps(summary), flush=True)
    if writer:
        writer.release()
    if log:
        log.close()
    if src:
        src.close()
    if intent_pub:
        intent_pub.close()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    v = sub.add_parser('verify'); v.add_argument('--skip-video', action='store_true'); v.add_argument('--device')
    for name in ('replay', 'robot', 'cameras'):
        p = sub.add_parser(name)
        if name == 'robot':
            p.add_argument('--host', required=True, help='robot camera server host (runs ComposedCameraServer)')
            p.add_argument('--port', type=int, default=5555)
            p.add_argument('--person-key', default='head', help='image key of the RGB camera looking at the person')
            p.add_argument('--hand-key', default='ego_view', help='image key of the RGB stream of the down-facing RGB-D camera')
        else:
            p.add_argument('--person', required=True, help='person camera: video file, USB index or URL')
            p.add_argument('--hand', required=True, help='down-facing camera RGB: video file, USB index or URL')
        p.add_argument('--device', help='cuda / mps / cpu (default: auto)')
        p.add_argument('--threshold', type=probability, default=None,
                       help='base minimum confidence for all classes (checkpoint default: 0.40)')
        for label in CLASSES:
            flag = label.replace('_', '-')
            p.add_argument(f'--{flag}-min-confidence', type=probability, default=None,
                           help=f'override minimum confidence only for {label}')
            p.add_argument(f'--{flag}-stable-frames', type=positive_int, default=None,
                           help=f'consecutive valid decisions required for {label} (default: 2)')
        p.add_argument('--max-skew', type=float, default=.1, help='max |t_hand - t_person| s; beyond it the hand frame is ignored')
        p.add_argument('--show', action='store_true'); p.add_argument('--save'); p.add_argument('--log')
        p.add_argument('--print', action='store_true', help='print every decision')
        p.add_argument('--intent-bind', help='publish versioned human-intent JSON, e.g. tcp://*:5562')
        p.add_argument('--hold-hug-on-unknown', action='store_true',
                       help='keep the last accepted hug across unknown decisions for a bounded time')
        p.add_argument('--hug-unknown-hold-seconds', type=float, default=3.0,
                       help='seconds to keep hug over unknown when --hold-hug-on-unknown is enabled')
        p.add_argument('--duration', type=float, default=None, help='stop after N seconds (live modes)')
    args = ap.parse_args()
    sys.exit(verify(args) if args.mode == 'verify' else live(args))


if __name__ == '__main__':
    main()
