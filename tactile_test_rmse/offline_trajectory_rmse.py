#!/usr/bin/env python3
"""Open-loop trajectory RMSE for the G1/BrainCo LeRobot test set.

The metric is global RMSE over every predicted scalar in every valid
action chunk.  Each query uses the recorded observation at t (open loop),
and compares the returned H-step chunk against recorded actions t:t+H.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DATASET = "/root/tactile_ws/datasets/test-dataset/test-dataset"
EMBODIMENT = "unitree_g1_sonic_brainco6dof"
UNIFOLM_TASK_NAME = "g1_sonic_brainco6dof"
ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
STATE_KEYS = (
    "left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"
)
VIDEO_KEYS = ("depth_camera_rgb", "webcam")
STATE_SLICES = {
    "left_leg": slice(0, 6), "right_leg": slice(6, 12), "waist": slice(12, 15),
    "left_arm": slice(15, 22), "left_hand": slice(22, 28), "right_arm": slice(28, 35),
    "right_hand": slice(35, 41),
}
ACTION_COLUMNS = {
    "motion_token": "action.motion_token",
    "left_hand_joints": "teleop.left_hand_joints",
    "right_hand_joints": "teleop.right_hand_joints",
}
ACTION_GROUPS = {
    "motion_token": slice(0, 64),
    "left_hand": slice(64, 70),
    "right_hand": slice(70, 76),
}
HORIZON_POSITIONS = (1, 10, 20, 40)  # human-readable, 1-indexed offsets


class EpisodeDataset:
    """Small LeRobot-v2 reader with no model-framework dependency."""

    def __init__(self, root: str):
        import pandas as pd

        self.root = Path(root)
        self.pd = pd
        try:
            from decord import VideoReader, cpu

            self.VideoReader = VideoReader
            self.cpu = cpu
            self.video_backend = "decord"
        except ImportError:
            # Isaac-GR00T's training container has torchcodec/OpenCV but no
            # decord; other local environments commonly have decord.  Both
            # paths return RGB uint8 arrays.
            import cv2

            self.cv2 = cv2
            self.video_backend = "opencv"
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()]
        self.tasks = {
            row["task_index"]: row["task"]
            for row in (json.loads(line) for line in (self.root / "meta/tasks.jsonl").read_text().splitlines())
        }
        self._videos: dict[tuple[int, str], Any] = {}

    def __len__(self) -> int:
        return len(self.episodes)

    def episode(self, episode_id: int):
        data_path = self.info["data_path"].format(episode_chunk=episode_id // self.info["chunks_size"], episode_index=episode_id)
        return self.pd.read_parquet(self.root / data_path)

    def frame(self, episode_id: int, key: str, step: int) -> np.ndarray:
        full_key = f"observation.images.{key}"
        cache_key = (episode_id, full_key)
        if cache_key not in self._videos:
            path = self.info["video_path"].format(
                episode_chunk=episode_id // self.info["chunks_size"],
                video_key=full_key,
                episode_index=episode_id,
            )
            if self.video_backend == "decord":
                self._videos[cache_key] = self.VideoReader(str(self.root / path), ctx=self.cpu(0))
            else:
                capture = self.cv2.VideoCapture(str(self.root / path))
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open video: {self.root / path}")
                self._videos[cache_key] = capture
        if self.video_backend == "decord":
            return self._videos[cache_key][step].asnumpy()
        capture = self._videos[cache_key]
        capture.set(self.cv2.CAP_PROP_POS_FRAMES, step)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {step} from {full_key}, episode {episode_id}")
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close_episode(self, episode_id: int) -> None:
        """Release all video decoder state belonging to one finished episode.

        Keeping one decoder per camera for every processed episode eventually
        consumes the whole host RAM (OpenCV/FFmpeg buffers are large).  The
        evaluator only revisits frames inside the current episode, so retaining
        older decoders has no benefit.
        """
        episode_keys = [key for key in self._videos if key[0] == episode_id]
        for key in episode_keys:
            reader = self._videos.pop(key)
            if self.video_backend == "opencv":
                reader.release()

    def step_data(
        self,
        episode_id: int,
        episode,
        step: int,
        horizon: int,
        video_delta_indices: tuple[int, ...] = (0,),
    ):
        from types import SimpleNamespace

        raw_state = np.asarray(episode["observation.state"].iloc[step], dtype=np.float32)
        states = {key: raw_state[STATE_SLICES[key]][None, :] for key in STATE_KEYS}
        # Some policies (notably this RLDX checkpoint) were trained with a
        # short image history.  Clamp negative indices at the first frame,
        # matching the usual edge-padding used by dataset processors.
        frame_indices = [
            min(max(step + delta, 0), len(episode) - 1)
            for delta in video_delta_indices
        ]
        images = {
            key: [self.frame(episode_id, key, frame_index) for frame_index in frame_indices]
            for key in VIDEO_KEYS
        }
        actions = {
            key: np.stack(episode[ACTION_COLUMNS[key]].iloc[step : step + horizon].to_numpy()).astype(np.float32)
            for key in ACTION_KEYS
        }
        task_id = int(np.asarray(episode["task_index"].iloc[step]).reshape(-1)[0])
        projected_gravity = np.asarray(
            episode["observation.projected_gravity"].iloc[step], dtype=np.float32
        )[None, :]
        return SimpleNamespace(
            states=states,
            projected_gravity=projected_gravity,
            images=images,
            actions=actions,
            text=self.tasks[task_id],
        )


def flat_action(actions: dict[str, Any]) -> np.ndarray:
    """Concatenate the three native action fields to (H, 76)."""
    values = []
    for key in ACTION_KEYS:
        value = actions.get(key, actions.get(f"action.{key}"))
        if value is None:
            raise KeyError(f"Policy did not return action.{key}; returned {list(actions)}")
        value = np.asarray(value, dtype=np.float32)
        # RLDX returns B,H,D; Being-H and UnifoLM return H,D.
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, received {value.shape}")
            value = value[0]
        if value.ndim == 1:
            value = value[:, None]
        values.append(value)
    return np.concatenate(values, axis=-1)


def rldx_predict(policy, data) -> np.ndarray:
    # This is the native RLDX wire shape: B,T,... for state/video and B,T,D
    # for action chunks returned by the policy.
    observation: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for key, value in data.images.items():
        observation["video"][key] = np.asarray(value)[None, :]
    for key, value in data.states.items():
        observation["state"][key] = np.asarray(value, dtype=np.float32)[None, :]
    observation["language"]["annotation.human.task_description"] = [[data.text]]
    actions, _ = policy.get_action(observation)
    return flat_action(actions)


def groot_predict(policy, data) -> np.ndarray:
    """GR00T N1.6 uses the same B,T,D observation/action wire format."""
    observation: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for key, value in data.images.items():
        observation["video"][key] = np.asarray(value)[None, :]
    for key, value in data.states.items():
        observation["state"][key] = np.asarray(value, dtype=np.float32)[None, :]
    observation["state"]["projected_gravity"] = data.projected_gravity[None, :]
    observation["language"]["annotation.human.task_description"] = [[data.text]]
    actions, _ = policy.get_action(observation)
    return flat_action(actions)


def beingh_predict(policy, data) -> np.ndarray:
    observation: dict[str, Any] = {}
    # Being-H detects batching from the first observation.  Keep every
    # modality consistently batched (B=1): video (B,H,W,C), state (B,D),
    # instruction list (B).  Mixing an unbatched state with a batched image
    # makes its internal packer receive a 1-D state tensor.
    for key, value in data.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for key, value in data.states.items():
        observation[f"state.{key}"] = np.asarray(value, dtype=np.float32)
    observation["language.instruction"] = [data.text]
    return flat_action(policy.get_action(observation))


def unifolm_predict(url: str, data) -> np.ndarray:
    import requests

    # The server's `json_numpy.patch()` decoder accepts the compact encoded
    # representation below. Keeping it in a single field avoids relying on
    # a client-side FastAPI/numpy JSON hook.
    try:
        import json_numpy
    except ImportError as exc:
        raise RuntimeError("UnifoLM evaluation needs json-numpy in the active environment") from exc

    state = np.concatenate([np.asarray(data.states[k], dtype=np.float32)[0] for k in STATE_KEYS])
    payload = {
        "observations": [
            {
                "full_image": np.asarray(data.images["depth_camera_rgb"])[0],
                "wrist_image": np.asarray(data.images["webcam"])[0],
                "state": state,
                "instruction": data.text,
                # This selects the normalization statistics stored in the
                # UnifoLM checkpoint.  It is not the RLDX/GR00T embodiment tag.
                "task_name": UNIFOLM_TASK_NAME,
            }
        ]
    }
    response = requests.post(url, json={"encoded": json_numpy.dumps(payload)}, timeout=180)
    response.raise_for_status()
    response_payload = response.json()
    if response_payload == "error":
        raise RuntimeError("UnifoLM server returned 'error'; inspect its server log")
    result = (
        json_numpy.loads(response_payload)
        if isinstance(response_payload, str)
        else response_payload
    )
    predicted = np.asarray(result, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape[-1] != 76:
        raise ValueError(f"Unexpected UnifoLM response shape {predicted.shape}")
    return predicted


def create_predictor(args):
    if args.backend == "rldx":
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy

        policy = RLDXPolicy(
            embodiment_tag=EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF,
            model_path=args.checkpoint,
            device=args.device,
        )
        return lambda data: rldx_predict(policy, data)

    if args.backend == "beingh":
        from BeingH.inference.beingh_policy import BeingHPolicy

        policy = BeingHPolicy(
            model_path=args.checkpoint,
            data_config_name="g1_sonic_brainco6dof",
            dataset_name="g1_sonic_brainco6dof_posttrain",
            embodiment_tag=EMBODIMENT,
            instruction_template="According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?",
            device=args.device,
            enable_rtc=False,
        )
        return lambda data: beingh_predict(policy, data)

    if args.backend == "groot":
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=args.device,
        )
        return lambda data: groot_predict(policy, data)

    return lambda data: unifolm_predict(args.unifolm_url, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rldx", "beingh", "groot", "unifolm"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unifolm-url", default="http://127.0.0.1:8777/act")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not Path(args.dataset).is_dir():
        raise FileNotFoundError(args.dataset)

    video_delta_indices = (0,)
    if args.backend == "rldx":
        config_path = Path(args.checkpoint) / "config.json"
        config = json.loads(config_path.read_text())
        video_length = int(config.get("video_length", 1))
        video_stride = int(config.get("video_stride", 1))
        video_delta_indices = tuple(
            (index - video_length + 1) * video_stride
            for index in range(video_length)
        )
        print(f"rldx_video_delta_indices={video_delta_indices}", flush=True)

    predictor = create_predictor(args)
    dataset = EpisodeDataset(args.dataset)
    episode_count = len(dataset) if args.max_episodes == 0 else min(args.max_episodes, len(dataset))
    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    scalar_count = 0
    absolute_error_blocks = []
    group_sse = {name: 0.0 for name in ACTION_GROUPS}
    group_count = {name: 0 for name in ACTION_GROUPS}
    horizon_sse = {position: 0.0 for position in HORIZON_POSITIONS}
    horizon_count = {position: 0 for position in HORIZON_POSITIONS}
    per_episode = []
    partial_output = Path(f"{args.output}.partial.json")

    for episode_id in range(episode_count):
        try:
            episode = dataset.episode(episode_id)
            # Do not pad action ground truth: chunks near the end are compared only
            # over available future frames.
            last_start = max(0, len(episode) - args.horizon + 1)
            ep_sse = 0.0
            ep_count = 0
            for step in range(0, last_start, args.horizon):
                data = dataset.step_data(
                    episode_id,
                    episode,
                    step,
                    args.horizon,
                    video_delta_indices=video_delta_indices,
                )
                target = np.concatenate([data.actions[k] for k in ACTION_KEYS], axis=-1)
                prediction = predictor(data)
                usable = min(len(target), len(prediction), args.horizon)
                prediction = prediction[:usable]
                target = target[:usable]
                if prediction.shape != target.shape:
                    raise ValueError(
                        f"Episode {episode_id}, step {step}: predicted {prediction.shape}, target {target.shape}"
                    )
                squared_error = np.square(prediction.astype(np.float64) - target.astype(np.float64))
                absolute_error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
                ep_sse += float(squared_error.sum())
                ep_count += squared_error.size
                sum_absolute_error += float(absolute_error.sum())
                absolute_error_blocks.append(absolute_error.reshape(-1))
                for name, group_slice in ACTION_GROUPS.items():
                    group_error = squared_error[:, group_slice]
                    group_sse[name] += float(group_error.sum())
                    group_count[name] += group_error.size
                for position in HORIZON_POSITIONS:
                    index = position - 1
                    if index < usable:
                        position_error = squared_error[index]
                        horizon_sse[position] += float(position_error.sum())
                        horizon_count[position] += position_error.size
            if ep_count:
                sum_squared_error += ep_sse
                scalar_count += ep_count
                per_episode.append({"episode": episode_id, "rmse": math.sqrt(ep_sse / ep_count)})
            print(f"episode={episode_id} rmse={per_episode[-1]['rmse'] if ep_count else float('nan'):.8f}", flush=True)
            if scalar_count:
                partial_output.parent.mkdir(parents=True, exist_ok=True)
                partial_output.write_text(
                    json.dumps(
                        {
                            "backend": args.backend,
                            "checkpoint": args.checkpoint,
                            "episodes_completed": len(per_episode),
                            "episodes_total": episode_count,
                            "rmse_so_far": math.sqrt(sum_squared_error / scalar_count),
                            "mae_so_far": sum_absolute_error / scalar_count,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        finally:
            dataset.close_episode(episode_id)

    report = {
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "episodes": episode_count,
        "compared_scalars": scalar_count,
        "rmse": math.sqrt(sum_squared_error / scalar_count),
        "mae": sum_absolute_error / scalar_count,
        "p95_absolute_error": float(np.percentile(np.concatenate(absolute_error_blocks), 95)),
        "rmse_by_action_group": {
            name: math.sqrt(group_sse[name] / group_count[name]) for name in ACTION_GROUPS
        },
        "rmse_by_horizon_position": {
            f"t+{position}": math.sqrt(horizon_sse[position] / horizon_count[position])
            for position in HORIZON_POSITIONS
            if horizon_count[position]
        },
        "per_episode": per_episode,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    partial_output.unlink(missing_ok=True)
    print(json.dumps({k: report[k] for k in report if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
