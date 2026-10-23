"""Headless CLI for streaming RTMW3D sequence extraction from video."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .rtmw3d_runner import PROJECT_ROOT, RTMW3DResult, RTMW3DRunner

# The runner guarantees that the bundled RTMLib copy is imported.
from rtmlib import draw_skeleton


NUM_KEYPOINTS = 133
UPPER_LIMB_KEYPOINTS = np.asarray([5, 6, 7, 8, 9, 10], dtype=np.intp)
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10
MIN_VISIBLE_UPPER_LIMB_KEYPOINTS = 3


@dataclass(frozen=True)
class SelectedPerson:
    """One pose with independent detection and visibility signals."""

    keypoints_3d: np.ndarray
    keypoints_2d: np.ndarray
    scores: np.ndarray
    visibility: np.ndarray
    person_detected: bool
    selected_person_index: int
    selection_mean_confidence: float | None
    upper_body_visible_count: int
    upper_body_valid: bool
    left_wrist_valid: bool
    right_wrist_valid: bool


def _output_directory(path: Path) -> Path:
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"--output-dir must be inside {artifacts_root}; received {candidate}"
        ) from exc
    return candidate


def _empty_person() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float32),
        np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32),
        np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
        np.zeros((NUM_KEYPOINTS,), dtype=bool),
    )


def _as_people(array: np.ndarray, coordinates: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.size == 0:
        return np.empty((0, NUM_KEYPOINTS, coordinates), dtype=np.float32)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3 or value.shape[1:] != (NUM_KEYPOINTS, coordinates):
        raise RuntimeError(
            f"Unexpected {name} shape: {value.shape}; expected [N, "
            f"{NUM_KEYPOINTS}, {coordinates}]"
        )
    return value.astype(np.float32, copy=False)


def _as_scores(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.size == 0:
        return np.empty((0, NUM_KEYPOINTS), dtype=np.float32)
    if value.ndim == 1:
        value = value[None, ...]
    if value.ndim != 2 or value.shape[1] != NUM_KEYPOINTS:
        raise RuntimeError(
            f"Unexpected scores shape: {value.shape}; expected [N, {NUM_KEYPOINTS}]"
        )
    return value.astype(np.float32, copy=False)


def _select_primary_person(result: RTMW3DResult, threshold: float) -> SelectedPerson:
    keypoints_3d = _as_people(result.keypoints_3d, 3, "keypoints_3d")
    keypoints_2d = _as_people(result.keypoints_2d, 2, "keypoints_2d")
    scores = _as_scores(result.scores)

    # Detection does not depend on mean confidence: with partial visibility,
    # legs may be absent even though the person bbox and upper body are valid.
    arrays_are_nonempty = (
        keypoints_3d.size > 0 and keypoints_2d.size > 0 and scores.size > 0
    )
    if result.num_people <= 0 or not arrays_are_nonempty:
        empty_3d, empty_2d, empty_scores, empty_visibility = _empty_person()
        return SelectedPerson(
            keypoints_3d=empty_3d,
            keypoints_2d=empty_2d,
            scores=empty_scores,
            visibility=empty_visibility,
            person_detected=False,
            selected_person_index=-1,
            selection_mean_confidence=None,
            upper_body_visible_count=0,
            upper_body_valid=False,
            left_wrist_valid=False,
            right_wrist_valid=False,
        )

    people = keypoints_3d.shape[0]
    if keypoints_2d.shape[0] != people or scores.shape[0] != people:
        raise RuntimeError(
            "RTMLib returned an inconsistent number of people: "
            f"3D={people}, 2D={keypoints_2d.shape[0]}, scores={scores.shape[0]}"
        )
    if int(result.num_people) != people:
        raise RuntimeError(
            "RTMLib returned inconsistent num_people and arrays: "
            f"num_people={result.num_people}, arrays={people}"
        )

    # With multiple boxes, select the person whose shoulders, elbows, and
    # wrists are most reliably visible. NaN values must not affect ranking.
    upper_limb_scores = scores[:, UPPER_LIMB_KEYPOINTS]
    safe_upper_limb_scores = np.where(
        np.isfinite(upper_limb_scores), upper_limb_scores, 0.0
    )
    mean_upper_limb_confidence = safe_upper_limb_scores.mean(axis=1)
    selected_index = int(np.argmax(mean_upper_limb_confidence))
    selected_confidence = float(mean_upper_limb_confidence[selected_index])

    # Preserve raw scores for the selected person. The threshold applies only
    # to the per-joint mask and does not discard the full pose under occlusion.
    selected_scores = scores[selected_index].copy()
    visibility = selected_scores >= threshold
    upper_body_visible_count = int(visibility[UPPER_LIMB_KEYPOINTS].sum())
    return SelectedPerson(
        keypoints_3d=keypoints_3d[selected_index].copy(),
        keypoints_2d=keypoints_2d[selected_index].copy(),
        scores=selected_scores,
        visibility=visibility,
        person_detected=True,
        selected_person_index=selected_index,
        selection_mean_confidence=selected_confidence,
        upper_body_visible_count=upper_body_visible_count,
        upper_body_valid=(
            upper_body_visible_count >= MIN_VISIBLE_UPPER_LIMB_KEYPOINTS
        ),
        left_wrist_valid=bool(visibility[LEFT_WRIST_INDEX]),
        right_wrist_valid=bool(visibility[RIGHT_WRIST_INDEX]),
    )


def _frame_time_seconds(
    capture: cv2.VideoCapture,
    source_frame_index: int,
    source_fps: float,
    timestamp_origin: float | None,
    previous_time: float | None,
) -> tuple[float, float | None, bool]:
    reported = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    used_fallback = False
    if math.isfinite(reported) and reported >= 0.0:
        if timestamp_origin is None:
            timestamp_origin = reported
        candidate = reported - timestamp_origin
    else:
        candidate = source_frame_index / source_fps
        used_fallback = True

    # Some OpenCV backends repeat POS_MSEC or always return zero. In that case,
    # use the frame_index/source_fps video timeline.
    if previous_time is not None and candidate <= previous_time:
        candidate = source_frame_index / source_fps
        used_fallback = True
    if previous_time is not None and candidate <= previous_time:
        candidate = previous_time + 1.0 / source_fps
        used_fallback = True
    return float(candidate), timestamp_origin, used_fallback


def _make_overlay(
    frame: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    person_detected: bool,
    upper_body_valid: bool,
    threshold: float,
    video_time_s: float,
) -> np.ndarray:
    overlay = frame.copy()
    if person_detected:
        # The overlay always uses pixel-space 2D coordinates, not 3D values.
        overlay = draw_skeleton(
            overlay,
            keypoints_2d[None, ...],
            scores[None, ...],
            openpose_skeleton=False,
            kpt_thr=threshold,
        )
    banner_text: str | None = None
    banner_color = (0, 0, 255)
    if not person_detected:
        banner_text = "no person detected"
    elif not upper_body_valid:
        banner_text = "low upper-body confidence"
        banner_color = (0, 165, 255)

    if banner_text is not None:
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            banner_text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            banner_color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"t={video_time_s:.3f}s",
        (12, overlay.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _latency_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _observed_sample_fps(video_times: np.ndarray) -> float | None:
    if video_times.size < 2:
        return None
    positive = np.diff(video_times)
    positive = positive[positive > 0]
    if positive.size == 0:
        return None
    median_dt = float(np.median(positive))
    return 1.0 / median_dt if median_dt > 0 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--det-model", type=Path)
    parser.add_argument("--pose-model", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be greater than zero")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in the range [0, 1]")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be greater than zero")

    output_dir = _output_directory(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses_path = output_dir / "poses.npz"
    metadata_path = output_dir / "metadata.json"
    overlay_path = output_dir / "overlay.mp4"
    if args.save_overlay and video_path == overlay_path.resolve():
        raise ValueError("The input video is overlay.mp4; overwriting is not allowed")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    writer: cv2.VideoWriter | None = None
    source_frame_indices: list[int] = []
    video_times: list[float] = []
    keypoints_3d_sequence: list[np.ndarray] = []
    keypoints_2d_sequence: list[np.ndarray] = []
    scores_sequence: list[np.ndarray] = []
    visibility_sequence: list[np.ndarray] = []
    latencies: list[float] = []
    person_detections: list[bool] = []
    upper_body_validity: list[bool] = []
    upper_body_visible_counts: list[int] = []
    left_wrist_validity: list[bool] = []
    right_wrist_validity: list[bool] = []
    selected_indices: list[int] = []
    selection_mean_confidences: list[float] = []
    warmup_latency_ms: float | None = None
    decoded_frames = 0
    timestamp_fallback_frames = 0

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise RuntimeError(f"Video reports an invalid FPS: {source_fps}")
        if source_frame_count <= 0:
            raise RuntimeError(
                f"Video reports an invalid frame count: {source_frame_count}"
            )
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError(
                f"Video reports an invalid resolution: {source_width}x{source_height}"
            )

        source_duration_s = source_frame_count / source_fps
        saved_sequence_fps = min(float(args.target_fps), source_fps)
        target_period_s = 1.0 / float(args.target_fps)
        sampling_tolerance_s = 0.5 / source_fps
        decode_limit = source_frame_count
        if args.max_seconds is not None:
            decode_limit = min(
                decode_limit, int(math.ceil(args.max_seconds * source_fps)) + 1
            )

        # A single runner creates detector and pose sessions once per video.
        runner = RTMW3DRunner(
            device=args.device,
            backend=args.backend,
            mode=args.mode,
            det_model=args.det_model,
            pose_model=args.pose_model,
        )

        next_target_time_s = 0.0
        timestamp_origin: float | None = None
        previous_video_time: float | None = None
        warmed_up = False

        with tqdm(total=decode_limit, desc="RTMW3D video", unit="frame") as progress:
            source_frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames += 1
                progress.update(1)
                video_time_s, timestamp_origin, used_fallback = _frame_time_seconds(
                    capture,
                    source_frame_index,
                    source_fps,
                    timestamp_origin,
                    previous_video_time,
                )
                previous_video_time = video_time_s
                if used_fallback:
                    timestamp_fallback_frames += 1

                if args.max_seconds is not None and video_time_s >= args.max_seconds:
                    break

                # Select frames by video time on the target grid rather than
                # using a fixed frame stride.
                if video_time_s + sampling_tolerance_s < next_target_time_s:
                    source_frame_index += 1
                    continue
                while next_target_time_s <= video_time_s + sampling_tolerance_s:
                    next_target_time_s += target_period_s

                # The first selected frame warms up the model, then the same
                # frame goes through measured inference and is included in T.
                if not warmed_up:
                    warmup_values = runner.warmup(frame, iterations=1)
                    warmup_latency_ms = warmup_values[0]
                    warmed_up = True

                result = runner.infer(frame)
                selected = _select_primary_person(result, args.threshold)

                source_frame_indices.append(source_frame_index)
                video_times.append(video_time_s)
                keypoints_3d_sequence.append(selected.keypoints_3d)
                keypoints_2d_sequence.append(selected.keypoints_2d)
                scores_sequence.append(selected.scores)
                visibility_sequence.append(selected.visibility)
                latencies.append(float(result.latency_ms))
                person_detections.append(selected.person_detected)
                upper_body_validity.append(selected.upper_body_valid)
                upper_body_visible_counts.append(selected.upper_body_visible_count)
                left_wrist_validity.append(selected.left_wrist_valid)
                right_wrist_validity.append(selected.right_wrist_valid)
                selected_indices.append(selected.selected_person_index)
                selection_mean_confidences.append(
                    np.nan
                    if selected.selection_mean_confidence is None
                    else selected.selection_mean_confidence
                )

                if args.save_overlay:
                    overlay = _make_overlay(
                        frame,
                        selected.keypoints_2d,
                        selected.scores,
                        selected.person_detected,
                        selected.upper_body_valid,
                        args.threshold,
                        video_time_s,
                    )
                    if writer is None:
                        height, width = overlay.shape[:2]
                        writer = cv2.VideoWriter(
                            str(overlay_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            saved_sequence_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError(
                                f"OpenCV could not create the overlay video: {overlay_path}"
                            )
                    writer.write(overlay)

                source_frame_index += 1

        if not keypoints_3d_sequence:
            raise RuntimeError("No video frames could be processed")

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    source_frame_index_array = np.asarray(source_frame_indices, dtype=np.int64)
    video_time_array = np.asarray(video_times, dtype=np.float64)
    keypoints_3d_array = np.stack(keypoints_3d_sequence).astype(np.float32, copy=False)
    keypoints_2d_array = np.stack(keypoints_2d_sequence).astype(np.float32, copy=False)
    scores_array = np.stack(scores_sequence).astype(np.float32, copy=False)
    visibility_array = np.stack(visibility_sequence).astype(bool, copy=False)
    latency_array = np.asarray(latencies, dtype=np.float64)
    person_detected_array = np.asarray(person_detections, dtype=bool)
    upper_body_valid_array = np.asarray(upper_body_validity, dtype=bool)
    upper_body_visible_count_array = np.asarray(
        upper_body_visible_counts, dtype=np.int16
    )
    left_wrist_valid_array = np.asarray(left_wrist_validity, dtype=bool)
    right_wrist_valid_array = np.asarray(right_wrist_validity, dtype=bool)
    selected_index_array = np.asarray(selected_indices, dtype=np.int32)
    selection_mean_confidence_array = np.asarray(
        selection_mean_confidences, dtype=np.float32
    )

    expected_shapes = {
        "keypoints_3d": (len(video_times), NUM_KEYPOINTS, 3),
        "keypoints_2d": (len(video_times), NUM_KEYPOINTS, 2),
        "scores": (len(video_times), NUM_KEYPOINTS),
        "visibility": (len(video_times), NUM_KEYPOINTS),
    }
    actual_shapes = {
        "keypoints_3d": keypoints_3d_array.shape,
        "keypoints_2d": keypoints_2d_array.shape,
        "scores": scores_array.shape,
        "visibility": visibility_array.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise RuntimeError(
                f"Final shape {name}={actual_shapes[name]}, expected {expected}"
            )

    # Preserve confidence and visibility with pseudo-label joints so the
    # downstream predictor does not interpret occlusion as coordinate (0,0,0).
    np.savez_compressed(
        poses_path,
        source_frame_index=source_frame_index_array,
        video_time_s=video_time_array,
        keypoints_3d=keypoints_3d_array,
        keypoints_2d=keypoints_2d_array,
        scores=scores_array,
        visibility=visibility_array,
        latency_ms=latency_array,
        person_detected=person_detected_array,
        upper_body_valid=upper_body_valid_array,
        upper_body_visible_count=upper_body_visible_count_array,
        left_wrist_valid=left_wrist_valid_array,
        right_wrist_valid=right_wrist_valid_array,
        selected_person_index=selected_index_array,
        selected_upper_limb_mean_confidence=selection_mean_confidence_array,
    )

    observed_sample_fps = _observed_sample_fps(video_time_array)
    frames_without_detection = int((~person_detected_array).sum())
    frames_low_upper_body_confidence = int(
        (person_detected_array & ~upper_body_valid_array).sum()
    )
    frames_invalid_left_wrist = int((~left_wrist_valid_array).sum())
    frames_invalid_right_wrist = int((~right_wrist_valid_array).sum())
    latency_stats = _latency_statistics(latency_array)
    metadata: dict[str, Any] = {
        "source_video": str(video_path),
        "source_fps": source_fps,
        "target_fps": float(args.target_fps),
        "saved_sequence_fps": saved_sequence_fps,
        "observed_median_sample_fps": observed_sample_fps,
        "source_duration_s": source_duration_s,
        "max_seconds": args.max_seconds,
        "source_frame_count": source_frame_count,
        "decoded_frame_count": decoded_frames,
        "processed_frame_count": len(video_times),
        "frames_without_detection": frames_without_detection,
        "frames_low_upper_body_confidence": frames_low_upper_body_confidence,
        "frames_invalid_left_wrist": frames_invalid_left_wrist,
        "frames_invalid_right_wrist": frames_invalid_right_wrist,
        "resolution": {"width": source_width, "height": source_height},
        "latency": latency_stats,
        "warmup_latency_ms": warmup_latency_ms,
        "versions": {
            "rtmlib": runner.rtmlib_version,
            "onnxruntime": runner.onnxruntime_version,
            "torch": runner.torch_version,
        },
        "available_onnxruntime_providers": list(runner.available_providers),
        "configuration": {
            "device": runner.device,
            "backend": runner.backend,
            "mode": runner.mode,
            "threshold": float(args.threshold),
            "det_model": None if runner.det_model is None else str(runner.det_model),
            "pose_model": None if runner.pose_model is None else str(runner.pose_model),
            "person_detection_rule": (
                "result.num_people > 0 and keypoints_3d/keypoints_2d/scores arrays "
                "are non-empty"
            ),
            "selection_rule": (
                "maximum mean confidence over upper-limb keypoints [5,6,7,8,9,10]"
            ),
            "upper_body_valid_rule": (
                "at least 3 visible keypoints among [5,6,7,8,9,10]"
            ),
        },
        "timestamp_source": "CAP_PROP_POS_MSEC with frame_index/source_fps fallback",
        "timestamp_fallback_frames": timestamp_fallback_frames,
        "overlay_saved": bool(args.save_overlay),
        "overlay_fps": saved_sequence_fps if args.save_overlay else None,
        "coordinate_note": (
            "keypoints_3d are camera/model-relative RTMW3D pseudo-labels, not "
            "global robot-frame coordinates; camera-to-robot extrinsic calibration "
            "is required"
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Video: {video_path}")
    print(
        f"Source: {source_width}x{source_height}, {source_fps:.6f} FPS, "
        f"{source_frame_count} frames, {source_duration_s:.3f} s"
    )
    print(
        f"Processed: {len(video_times)} frames at saved sequence FPS "
        f"{saved_sequence_fps:.6f}; without detection: {frames_without_detection}; "
        f"low upper-body confidence: {frames_low_upper_body_confidence}"
    )
    print(
        "Latency mean/median/p95: "
        f"{latency_stats['mean_ms']:.3f} / {latency_stats['median_ms']:.3f} / "
        f"{latency_stats['p95_ms']:.3f} ms"
    )
    print(f"poses.npz: {poses_path}")
    print(f"metadata.json: {metadata_path}")
    if args.save_overlay:
        print(f"overlay.mp4: {overlay_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
