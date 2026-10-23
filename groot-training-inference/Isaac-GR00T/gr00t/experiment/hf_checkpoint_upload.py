from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class HfCheckpointUploadCallback(TrainerCallback):
    """Synchronously upload saved checkpoints, then prune older local copies."""

    def __init__(self, repo_id: str, token_file: str, upload_every: int = 5000):
        if upload_every <= 0:
            raise ValueError("upload_every must be positive")
        self.repo_id = repo_id
        self.token_file = Path(token_file)
        self.upload_every = upload_every

    def _upload_and_prune(self, output_dir: str, step: int) -> None:
        checkpoint = Path(output_dir) / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("HF token file is empty")

        api = HfApi(token=token)
        api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(checkpoint),
            path_in_repo=checkpoint.name,
            commit_message=f"Upload GR00T checkpoint-{step}",
        )

        remote_files = set(api.list_repo_files(repo_id=self.repo_id, repo_type="model"))
        expected = {
            f"{checkpoint.name}/{path.relative_to(checkpoint).as_posix()}"
            for path in checkpoint.rglob("*")
            if path.is_file()
        }
        missing = expected - remote_files
        if missing:
            raise RuntimeError(
                f"HF verification failed for checkpoint-{step}: {len(missing)} files missing"
            )

        for candidate in Path(output_dir).iterdir():
            if not candidate.is_dir() or not candidate.name.startswith("checkpoint-"):
                continue
            try:
                candidate_step = int(candidate.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if candidate_step < step:
                shutil.rmtree(candidate)
                logger.info("Removed local checkpoint after verified HF upload: %s", candidate)

    def on_train_begin(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if state.is_world_process_zero and step > 0 and step % self.upload_every == 0:
            self._upload_and_prune(args.output_dir, step)
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if not state.is_world_process_zero or step % self.upload_every != 0:
            return control
        try:
            self._upload_and_prune(args.output_dir, step)
        except Exception as exc:
            logger.error(
                "HF upload failed at step %s (%s); stopping with local checkpoints retained",
                step,
                type(exc).__name__,
            )
            control.should_training_stop = True
            raise RuntimeError(
                f"HF checkpoint upload failed at step {step}; training stopped and local checkpoints retained"
            ) from None
        logger.info("Uploaded and verified checkpoint-%s to HF repo %s", step, self.repo_id)
        return control
