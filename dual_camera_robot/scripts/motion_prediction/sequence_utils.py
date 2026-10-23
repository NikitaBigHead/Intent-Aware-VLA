"""Shared safe IO and video-time indexing; no pose runtime imports."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_outputs(directory, names, overwrite=False):
    directory = Path(directory).expanduser().resolve()
    paths = [directory / name for name in names]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Output symlinks are not allowed')
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError('Use --overwrite for existing outputs: ' + ', '.join(existing))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('video_time_s must be nonempty, finite and strictly increasing')
    return times


def target_indices(times, origins, horizons):
    times = validate_times(times)
    desired = np.asarray(origins)[:, None] + np.asarray(horizons)[None, :]
    # Inclusive ceiling; tolerate floating-point noise at an exact sample time.
    indices = np.searchsorted(times, desired - 1e-9, side='left')
    return np.where(indices < len(times), indices, -1)


def find_episode(dataset_root, episode=357, camera='observation.images.webcam'):
    if episode < 0 or not camera or any(x in camera for x in ('/', '\\', '*', '?', '[', ']')):
        raise ValueError('Invalid episode or camera name')
    matches = sorted(Path(dataset_root).glob(f'videos/chunk-*/{camera}/episode_{episode:06d}.mp4'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one video for episode {episode}, found {matches}')
    return matches[0].resolve()


def latency_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'count': int(values.size), 'mean_ms': float(values.mean()) if values.size else None,
            'median_ms': float(np.median(values)) if values.size else None,
            'p95_ms': float(np.percentile(values, 95)) if values.size else None}
