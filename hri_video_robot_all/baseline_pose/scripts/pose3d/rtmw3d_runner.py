"""Reusable wrapper around the bundled RTMLib Wholebody3d implementation."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Import PyTorch before ONNX Runtime so ORT can use compatible CUDA/cuDNN
# runtime libraries shipped with the torch wheel.
import torch

try:
    import onnxruntime as ort
except ImportError:  # The OpenCV backend remains available without ONNX Runtime.
    ort = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RTMLIB_ROOT = PROJECT_ROOT / "3D-pose" / "rtmlib"
if not LOCAL_RTMLIB_ROOT.is_dir():
    raise ImportError(f"Bundled RTMLib not found: {LOCAL_RTMLIB_ROOT}")

# Prioritize the bundled copy to avoid importing another version from a venv.
local_rtmlib_path = str(LOCAL_RTMLIB_ROOT)
if local_rtmlib_path not in sys.path:
    sys.path.insert(0, local_rtmlib_path)

import rtmlib as _local_rtmlib  # noqa: E402
from rtmlib import Wholebody3d  # noqa: E402
from rtmlib.version import __version__ as rtmlib_version  # noqa: E402

rtmlib_module_path = Path(_local_rtmlib.__file__).resolve()
if not rtmlib_module_path.is_relative_to(LOCAL_RTMLIB_ROOT.resolve()):
    raise ImportError(
        "Imported RTMLib from an unexpected location: "
        f"{rtmlib_module_path}; expected a copy under {LOCAL_RTMLIB_ROOT}"
    )


@dataclass(frozen=True)
class RTMW3DResult:
    """Result of one measured RTMW3D call."""
    keypoints_3d: np.ndarray
    scores: np.ndarray
    keypoints_simcc: np.ndarray
    keypoints_2d: np.ndarray
    latency_ms: float
    num_people: int


class RTMW3DRunner:
    """One long-lived Wholebody3d instance for image, video, and live code."""

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "onnxruntime",
        mode: str = "balanced",
        det_model: str | Path | None = None,
        pose_model: str | Path | None = None,
    ) -> None:
        self.device = str(device)
        self.backend = str(backend).lower()
        self.mode = str(mode)
        self.det_model = self._validate_optional_model_path(det_model, "det_model")
        self.pose_model = self._validate_optional_model_path(pose_model, "pose_model")

        self.available_providers = self._get_available_providers()
        if self.backend == "onnxruntime" and ort is None:
            raise RuntimeError(
                "backend='onnxruntime' was requested, but onnxruntime or "
                "onnxruntime-gpu could not be imported."
            )
        if self.backend == "onnxruntime" and self.device.lower().startswith("cuda"):
            if "CUDAExecutionProvider" not in self.available_providers:
                providers = ", ".join(self.available_providers) or "no providers available"
                raise RuntimeError(
                    "RTMW3D on CUDA was requested, but ONNX Runtime does not "
                    f"provide CUDAExecutionProvider. Available providers: {providers}. "
                    "Install only onnxruntime-gpu and verify CUDA/cuDNN "
                    "compatibility with PyTorch."
                )

        if self.mode not in Wholebody3d.MODE:
            choices = ", ".join(sorted(Wholebody3d.MODE))
            raise ValueError(f"Unknown RTMW3D mode={self.mode!r}; available: {choices}")

        model_kwargs: dict[str, Any] = {
            "mode": self.mode,
            "backend": self.backend,
            "device": self.device,
        }
        if self.det_model is not None:
            model_kwargs["det"] = str(self.det_model)
        if self.pose_model is not None:
            model_kwargs["pose"] = str(self.pose_model)

        # Create the model and both ONNX Runtime sessions exactly once. Video
        # and live pipelines repeatedly call infer() on this instance.
        self._model = Wholebody3d(**model_kwargs)

    @staticmethod
    def _validate_optional_model_path(
        path: str | Path | None, argument_name: str
    ) -> Path | None:
        if path is None:
            return None
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Local model from {argument_name} was not found: {candidate}"
            )
        return candidate

    @staticmethod
    def _get_available_providers() -> tuple[str, ...]:
        if ort is None:
            return ()
        try:
            return tuple(str(provider) for provider in ort.get_available_providers())
        except Exception as exc:
            raise RuntimeError(
                f"Could not query ONNX Runtime providers: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a BGR numpy.ndarray, received {type(image).__name__}"
            )
        if image.size == 0:
            raise ValueError("image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape (height, width, 3) with BGR channel order; "
                f"received shape {image.shape}"
            )
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"image must have a numeric dtype, received {image.dtype}")
        return np.ascontiguousarray(image)

    @staticmethod
    def _to_numpy(value: Any, name: str) -> np.ndarray:
        if value is None:
            raise TypeError(f"RTMLib returned None instead of the {name} array")
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        try:
            return np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"Could not convert the {name} result to NumPy: {exc}"
            ) from exc

    def infer(self, image: np.ndarray) -> RTMW3DResult:
        """Run one synchronous inference call on a non-empty BGR image."""
        frame = self._validate_image(image)
        started = time.perf_counter()
        raw_result = self._model(frame)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(raw_result, (tuple, list)) or len(raw_result) != 4:
            actual = len(raw_result) if isinstance(raw_result, (tuple, list)) else type(raw_result).__name__
            raise RuntimeError(
                "Bundled RTMLib 0.0.16 must return exactly four values "
                "(keypoints_3d, scores, keypoints_simcc, keypoints_2d); "
                f"received: {actual}"
            )

        keypoints_3d_raw, scores_raw, keypoints_simcc_raw, keypoints_2d_raw = raw_result
        keypoints_3d = self._to_numpy(keypoints_3d_raw, "keypoints_3d")
        scores = self._to_numpy(scores_raw, "scores")
        keypoints_simcc = self._to_numpy(keypoints_simcc_raw, "keypoints_simcc")
        keypoints_2d = self._to_numpy(keypoints_2d_raw, "keypoints_2d")

        if keypoints_3d.size == 0:
            num_people = 0
        elif keypoints_3d.ndim >= 3:
            num_people = int(keypoints_3d.shape[0])
        else:
            num_people = 1

        # Store confidence with joints so a reliable estimate can be
        # distinguished from occlusion or a missing joint.
        return RTMW3DResult(
            keypoints_3d=keypoints_3d,
            scores=scores,
            keypoints_simcc=keypoints_simcc,
            keypoints_2d=keypoints_2d,
            latency_ms=latency_ms,
            num_people=num_people,
        )

    def warmup(self, image: np.ndarray, iterations: int = 1) -> tuple[float, ...]:
        """Warm up sessions and return latency for every unrecorded pass."""
        if iterations < 0:
            raise ValueError("iterations cannot be negative")
        frame = self._validate_image(image)
        return tuple(self.infer(frame).latency_ms for _ in range(iterations))

    @property
    def rtmlib_version(self) -> str:
        return str(rtmlib_version)

    @property
    def onnxruntime_version(self) -> str | None:
        return None if ort is None else str(ort.__version__)

    @property
    def torch_version(self) -> str:
        return str(torch.__version__)
