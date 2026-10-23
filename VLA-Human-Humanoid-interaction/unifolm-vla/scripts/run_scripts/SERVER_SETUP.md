# GPU server setup: UnifoLM-VLA 40k run

Source: `/workspace/project/extended-sep-24-with-interact-60` (1234 episodes).
With seed 42, 638 active episodes became `interact`: 170 fist bumps,
239 handshakes and 229 hugs. All 171 `none` episodes are unchanged.
The original dataset is untouched. All 1234 parquet labels and both video
frame counts were checked.

## RLDS conversion

`prepare_training_data.sh` streams directly from LeRobot Parquet and the two MP4
views into RLDS. Frames are resized to 224×224 before JPEG encoding. It uses
`/dev/shm` for TFDS shuffle files and writes the final dataset under
`/workspace/project/uniformvla_work/rlds`. The full build completed on
2026-09-26. TFDS reports 1234 episodes, 342375 frames and 12.49 GiB of
records (13 GiB on disk). Reading every episode through TFDS gave the task
counts: interact 638, none 171, handshake 159, hug 152, fist_bump 114.

The training and TFDS environment is at
`/workspace/project/uniformvla_work/.venv-tfds` (TensorFlow 2.15, TFDS 4.9.3,
PyTorch 2.5.1, Transformers 4.52.3, Accelerate 1.5.2).
To rebuild the full conversion if needed, run:

```bash
cd /workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla
bash scripts/run_scripts/prepare_training_data.sh
```

The builder can also use its earlier HDF5 mode when
`LEROBOT_DATASET_ROOT` is unset and `HDF5_OUTPUT_DIR` is set.

## Training

The 1×A100 launcher runs 40,000 steps and writes full model checkpoints at
30,000 and 40,000 to `/workspace/project/uniformvla_chpts`. `final_model/pytorch_model.pt`
is a hard link to the 40,000-step checkpoint, avoiding a third copy.

The local `Unifolm-VLM-Base` weights are at
`/workspace/project/uniformvla_work/Unifolm-VLM-Base`. The launcher checks for a
free GPU and at least 90 GiB free at the checkpoint path. Training uses
single-GPU Accelerate with bf16 and PyTorch SDPA.

Start in tmux from any machine with the SSH alias:

```bash
ssh gpu-server-b 'bash /workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla/scripts/run_scripts/start_40k_tmux.sh'
```

Monitor with `ssh gpu-server-b 'tmux attach -t uniformvla-40k'` or
`ssh gpu-server-b 'tail -f /workspace/project/uniformvla_chpts/extended_sep24_interact60_40k.log'`.
