# UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family
 <p style="font-size: 1.2em;">
    <a href="https://unigen-x.github.io/unifolm-vla.github.io"><strong>Project Page</strong></a> | 
    <a href="https://huggingface.co/unitreerobotics/models"><strong>Models</strong></a> |
    <a href="https://huggingface.co/unitreerobotics/datasets"><strong>Datasets</strong></a> 
  </p>
<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>

**UnifoLM-VLA-0** is a Vision–Language–Action (VLA) large model in the UnifoLM series, designed for general-purpose humanoid robot manipulation. It goes beyond the limitations of conventional Vision–Language Models (VLMs) in physical interaction. Through continued pre-training on robot manipulation data, the model evolves from "vision-language understanding" to an "embodied brain" equipped with physical common sense.

<table width="100%">
  <tr>
    <th width="50%">Spatial Semantic Enhancement</th>
    <th width="50%">Manipulation Generalization</th>
  </tr>
  <tr>
    <td valign="top">
      To address the requirements for instruction comprehension and spatial understanding in manipulation tasks, the model deeply integrates textual instructions with 2D/3D spatial details through continued pre-training, <strong>substantially strengthening its spatial perception and geometric understanding capabilities.</strong>
    </td>
    <td valign="top">
      By leveraging full dynamics prediction data, the model achieves strong generalization across diverse manipulation tasks. In real-robot validation, <strong>it can complete 12 categories of complex manipulation tasks with high quality using only a single policy.</strong>
    </td>
  </tr>
</table>



<div align="center">
  <img 
    src="assets/gif/UnifoLM-VLA-0.gif"
    style="width:100%; max-width:1000px; height:auto;"
  />
</div>

## 🔥 News
* Jan 29, 2026: 🚀 We released the training and inference code, along with the model weights for [**UnifoLM-VLA-0**](https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c).

## 📑 Open-Source Plan
- [x] Training 
- [x] Inference
- [x] Checkpoints

## ⚙️  Installation
This project is built on **CUDA 12.4**, and using the same version is strongly recommended to ensure compatibility.
```
conda create -n unifolm-vla python==3.10.18
conda activate unifolm-vla

git clone https://github.com/unitreerobotics/unifolm-vla.git

# If you already downloaded the repo:
cd unifolm-vla
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68"
pip install -e .

# Install FlashAttention2
pip install "flash-attn==2.5.6" --no-build-isolation

```
## 🧰 Model Checkpoints
| Model | Description | Link|
|---------|-------|------|
|`UnifoLM-VLM-Base` | Fine-tuned on general-purpose image–text VQA data and open-source robot datasets. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base)|
|`UnifoLM-VLA-Base` | Fine-tuned on [Unitree opensource](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab) dataset. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Base)|
|`UnifoLM-VLA-LIBERO`| Fine-tuned on [Libero](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab) dataset. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero)|

## 🛢️ Dataset
In our experiments, we consider the following twelve open-source dataset:
| Dataset | Robot | Link |
|---------|-------|------|
|G1_Stack_Block| [Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Stack_Block)|
|G1_Bag_Insert|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Bag_Insert)|
|G1_Erase_Board|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Erase_Board)|
|G1_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table)|
|G1_Pack_PencilBox|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PencilBox)|
|G1_Pour_Medicine|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine)|
|G1_Pack_PingPong|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong)|
|G1_Prepare_Fruit|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit)|
|G1_Organize_Tools|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools)|
|G1_Fold_Towel|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel)|
|G1_Wipe_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table)|
|G1_DualRobot_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_DualRobot_Clean_Table)|

To train on your own dataset, ensure the data follows the [Huggingface LeRobot V2.1](https://github.com/huggingface/lerobot) dataset format. Assume the source directory structure of the dataset is as follows:
```
source_dir/
    ├── dataset1_name
    ├── dataset2_name
    ├── dataset3_name
    └── ...
```
Then, run the following command to convert the dataset from LeRobot format to HDF5 format:
```python
cd prepare_data
python convert_lerobot_to_hdf5.py \
    --data_path /path/to/your/source_dir/dataset1_name \
    --target_path /path/to/save/the/converted/data/directory
```
Finally, run the following command to convert the HDF5 format into the RLDS dataset format required for training. Be sure to update the path ([here](prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py#L232)) to the correct location of the HDF5 data.
```
cd prepare_data/hdf5_to_rlds/rlds_dataset
tfds build  --data_dir  /path/to/save/the/converted/data/directory
```
The directory structure of the converted RLDS dataset is as follows:
```
source_dir/
├── downloads
├── rlds_dataset
         └── 1.0.0
```
The `1.0.0` directory is the final RLDS dataset version that can be used for training. The final directory should be kept as `source_dir/1.0.0`(e.g., `g1_stack_block/1.0.0`).

## 🚴‍♂️ Training
To train on a single dataset or multiple datasets, follow the steps below:
- **Step 1**: Assuming you have already prepared the RLDS dataset, register the dataset (e.g., the Unitree open-source dataset `G1_StackBox`) with our dataloader by adding an entry for it in `configs.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py#L58)), `transforms.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py#L948)), and `mixtures.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py#L366)) and `datasets.py`([here](src/unifolm_vla/rlds_dataloader/datasets/datasets.py#L106)). For reference, in each of these files, there are sample entries for the G1 datasets that we used in experiments.
- **Step 2**: Before starting fine-tuning, configure the size of the action chunks predicted by the model, the action and state degrees of freedom in the dataset, and the data normalization scheme in `constants.py` ([here](src/unifolm_vla/rlds_dataloader/constants.py#L70)). Refer to `NUM_ACTIONS_CHUNK`, `ACTION_DIM`, `PROPRIO_DIM`, and `ACTION_PROPRIO_NORMALIZATION_TYPE` in `G1_CONSTANTS`.
- **Step 3**:  please complete the configuration in the following order (see [here](scripts/run_scripts/run_unifolm_vla_train.sh)):
1. **Model Initialization**: Set `base_vlm` to the local path or the corresponding model weight URL of **UnifoLM-VLM-Base**, which will be used to initialize the vision–language backbone model.
2. **Dataset Path Configuration**: After configuring the model path, set `oxe_data_root` to the root directory of the dataset to ensure that the training script can correctly load the RLDS data.
3. **Dataset Mixture Specification**: Based on the configured data root, set `data_mix` to the name of the dataset(s) to be used for training or to the desired dataset mixture.
4. **Model Checkpoint Saving**: Specify the paths for saving model checkpoints and logs, which will store the model weights and training states generated during fine-tuning for later recovery, evaluation, and inference.
5. **Parallelism Configuration**: Finally, adjust `num_processes` according to the number of available GPUs to match the scale of distributed training.
- **Step 4**: You can now start fine-tuning. Before running the script [`run_unifolm_vla_train.sh`](scripts/run_scripts/run_unifolm_vla_train.sh),

## 🧩 Custom Recipe: G1 + BrainCo Revo2, SONIC Motion-Token Fine-Tune

This fork adds a ready-made recipe for fine-tuning on a Unitree G1 +
BrainCo Revo2 (6-DOF hand) dataset ("sep-10-2026-cleaned-merged", 535
episodes / ~182k frames / 50 Hz, teleop via NVlabs GR00T-WholeBodyControl's
GEAR-SONIC whole-body controller), where the action target is a **64-dim
SONIC motion-token latent** plus both hands' joint commands, instead of raw
joint targets. Everything below was written and reviewed without a GPU or a
local copy of this dataset (both were unavailable while this was put
together) -- see each file's own comments for exactly what is a verified
fact (e.g. read from this repo's own source) vs. an assumption taken from
the dataset's schema description, flagged inline wherever that distinction
matters.

### Action target

76-dim continuous vector, predicted as a 40-step chunk (0.8s lookahead @ 50Hz):
- 64 dims: `action.motion_token` (SONIC latent, encoded live during teleop)
- 6 dims: `teleop.left_hand_joints`
- 6 dims: `teleop.right_hand_joints` -- exactly 0.0 across every episode in
  the *original* 535-episode dataset ("sep-10-2026-cleaned-merged"). **CORRECTED**
  (task owner, after the dataset grew to ~1000 episodes with new hug/handshake/
  fist-bump tasks): that all-zero-ness was not purely an artifact of the old
  task set. In the new tasks the right ARM does move, but through
  `action.motion_token` / the proprio state fields, not through this field --
  the right hand's own DOF range is hardware-limited and reportedly cannot
  close into a full fist or a functional handshake grip even there. So this
  field may still be a narrow/near-degenerate signal in the expanded dataset;
  don't assume it's exactly 0.0 for the new data, but don't assume it spans a
  full functional range either -- re-profile real min/max/std once the new
  dataset's stats exist. Always included regardless; see constants.py's
  comment for why dropping it would desync ACTION_DIM, and for a related
  BOUNDS_Q99 normalization-risk note this correction raises
- (`action.wbc`, the 41-dim raw joint-target field also present in this
  dataset, is intentionally **not** used -- debug/reference field only)

Proprio/state is 41-dim (legs 12 + waist 3 + left arm+hand 13 + right
arm+hand 13), read straight from `observation.state`. Only 2 of the
dataset's 3 camera streams are used -- `depth_camera_rgb` (RGB despite the
name) and `webcam`; `external_view_camera` is excluded end-to-end (never
decoded by the HDF5 conversion step, never referenced by the RLDS schema).
No validation split -- all 535/535 episodes go to train.

### What's wired up in this fork

unifolm-vla registers new datasets very differently from a LeRobot-parquet
reader: LeRobot v2.1 -> `prepare_data/convert_lerobot_to_hdf5*.py` -> HDF5 ->
`prepare_data/hdf5_to_rlds/rlds_dataset*/` (`tfds build`) -> a versioned
RLDS/TFDS dataset, which is *then* registered into the OXE dataloader via 4
files. See each file's own header/inline comments for full reasoning:

- **Conversion (new files, not in-place patches)**:
  `prepare_data/convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py` and
  `prepare_data/hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof/rlds_dataset.py`.
  New files rather than edits to the checked-in `convert_lerobot_to_hdf5.py`
  / `rlds_dataset/rlds_dataset.py` because those are themselves already
  hardcoded, per-dataset scripts for a *different*, unrelated G1 dataset
  (dual-arm EE poses, `cam_left_high`/`cam_right_high` camera keys) -- every
  one of the README's 12 open-source datasets was evidently converted the
  same way, with its own private script copy; only one generic example of
  each is checked into the repo. The conversion script excludes
  `external_view_camera` by mutating `LeRobotDataset.meta.video_keys` before
  any frame is read (verified against `LeRobotDataset`'s source at the exact
  commit this repo's own README installs -- there is no constructor argument
  to do this, so this was the only correct way found).
- **OXE/RLDS registration** (existing files, patched): `oxe/configs.py` (new
  `g1_sonic_brainco6dof` entry + new `StateEncoding`/`ActionEncoding` enum
  members), `oxe/transforms.py` (new pass-through standardize_fn),
  `oxe/mixtures.py` (new single-dataset mixture), `datasets/datasets.py`
  (explicit camera-view branch, mirroring the existing G1 entries as
  templates per the upstream README). Also `oxe/materialize.py` -- **not**
  one of the four files the upstream README names, but required: it
  whitelists which `ActionEncoding` values are loadable at all and defines
  their `absolute_action_mask`/`action_normalization_mask`, and raises a
  `ValueError` for any encoding not on that list.
- **Constants**: `rlds_dataloader/constants.py`'s new
  `G1_SONIC_BRAINCO6DOF_CONSTANTS` (`ACTION_DIM=76`, `PROPRIO_DIM=41`,
  `NUM_ACTIONS_CHUNK=40`, `ACTION_PROPRIO_NORMALIZATION_TYPE=BOUNDS_Q99`).
  **Read the normalization-type comment there before changing it** -- this
  dataset's `motion_token` is a SONIC latent with no known hard physical
  range (unlike the existing G1 entries' joint/EEF values), so `BOUNDS_Q99`
  (robust to outlier frames) was chosen over plain `BOUNDS` (min/max, which
  one outlier frame can silently crush); this has not been checked against
  the real per-dimension statistics.
- **Image augmentation**: wired into the *existing* hook (`RLDSDataset.__init__`
  in `datasets/datasets.py`, dlimp's `augment_image`) rather than a new one --
  `image_aug` was previously hardcoded `False` in `train_unifolm_vla.py`
  regardless of config, now reads `cfg.trainer.image_aug`. The requested
  ColorJitter/RandomResizedCrop params are approximated onto dlimp's
  parameterization as closely as the hook allows; **random rotation could
  not be included** -- there is no rotation op anywhere in this pipeline's
  only augmentation hook, and per instructions this adaptation does not
  hand-roll a new one. No horizontal flip, by construction (preserves
  left/right hand action semantics).

### How to run

```bash
# Stage 1-3: convert the dataset (LeRobot -> HDF5 -> RLDS/TFDS). Prints and
# runs stage 1, prints (but does not run) the stage-3 tfds build command --
# see the script for why:
LEROBOT_DATASET_ROOT=/path/to/lerobot/sep-10-2026-cleaned-merged \
HDF5_OUTPUT_DIR=/path/to/save/hdf5 \
OXE_DATA_ROOT=/path/to/save/rlds \
  bash scripts/run_scripts/convert_g1_sonic_brainco6dof_dataset.sh

# Full fine-tune (1x A100 80GB, bf16, 40k steps):
OXE_DATA_ROOT=/path/to/save/rlds \
BASE_VLM=/path/to/your/Unifolm-VLM-Base \
  bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh
```

The training script prints the exact resolved `accelerate launch` command
before running it. Open it for the full VRAM-budget reasoning (the ~7.62B
text decoder + lm_head stay frozen -- full-parameter fine-tuning of the whole
~8.3B backbone does not fit in 80GB without ZeRO sharding across multiple
GPUs -- but the ~676M-param ViT is unfrozen alongside the flow-matching
action head, per a later revision motivated by this dataset's close-proximity
hug/handshake/fist-bump visual domain; see the script's own comment for the
module-tree trace that made vision-only unfreezing possible and the updated
VRAM math) and for a list of several
config knobs in this codebase that turned out to be silently dead
(`trainer.warmup_ratio`, `trainer.weight_decay`,
`trainer.gradient_accumulation_steps`, `trainer.enable_gradient_checkpointing`)
discovered while preparing this recipe.

Both the dataset conversion and the fine-tune can run for hours — launch
each inside `tmux`/`screen` so it survives a dropped connection:

```bash
tmux new-session -d -s unifolm-g1-convert -c "$(pwd)" \
  "LEROBOT_DATASET_ROOT=/path/to/lerobot/sep-10-2026-cleaned-merged HDF5_OUTPUT_DIR=/path/to/save/hdf5 OXE_DATA_ROOT=/path/to/save/rlds bash scripts/run_scripts/convert_g1_sonic_brainco6dof_dataset.sh; exec bash"

tmux new-session -d -s unifolm-g1-ft -c "$(pwd)" \
  "OXE_DATA_ROOT=/path/to/save/rlds BASE_VLM=/path/to/your/Unifolm-VLM-Base bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```

## 🌏 Simulation Inference Evaluation
To evaluate the **UnifoLM-VLA-Libero** model in the `LIBERO` simulation environment ([here](https://huggingface.co/datasets/openvla/modified_libero_rlds)), follow the steps below:
- **Step 1**: Install the LIBERO simulation environment and its dependencies::
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/LIBERO/libero_requirements.txt  # Run from the UnifoLM-VLA project root directory
```
- **Step 2**: In `run_eval_libero.sh` ([here](scripts/eval_scripts/run_eval_libero.sh)), modify the following fields:  ```your_ckpt```, ```task_suite_name```, ```unnorm_key```, and ```LIBERO_HOME``` and  ```vlm_pretrained_path```.
- **Step 3**: Launch the evaluation:
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_eval_libero.sh
```

## 🤖 Real-World Inference Evaluation

In our system, inference is executed on the server side. The robot client collects observations from the real robot and sends them to the server for action inference. The full pipeline can be completed by following the steps below.

### Server Setup
- **Step 1**: In `run_real_eval_server.sh` ([here](scripts/eval_scripts/run_real_eval_server.sh)), modify the following fields:  ```ckpt_path```, ```port```, and ```unnorm_key``` and  ```vlm_pretrained_path```.
- **Step 2**: Launch the server:
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh
```

### Client Setup

- **Step 1**: Refer to [unitree_deploy/README.md](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/unitree_deploy/README.md) to create the ```unitree_deploy``` conda environment, install the required dependencies, and start the controller or service on the real robot.

- **Step 2**: Open a new terminal and establish a tunnel connection from the client to the server:
```
ssh user_name@remote_server_IP -CNg -L port:127.0.0.1:port
```
- **Step 3**: Modify and run the script ```unitree_deploy/robot_client.py``` as a reference.

## 📝 Codebase Architecture
Here's a high-level overview of the project's code structure and core components:
```
unifolm-vla/
    ├── assets                      # Media assets such as GIFs
    ├── experiments                 # Libero datasets for running inference
    ├── deployment                  # Deployment server code
    ├── prepare_data                # Scripts for dataset preprocessing and format conversion
    ├── scripts                     # Main scripts for training, evaluation, and deployment
    ├── src
    │    ├──unifolm_vla             # Core Python package for the Unitree world model
    │    │      ├── config          # Configuration files for training
    │    │      ├── model           # Model architectures and backbone definitions
    │    │      ├── rlds_dataloader # Dataset loading, transformations, and dataloaders
    │    │      └── training        # Model Training
```

## 🙏 Acknowledgement
Lots of code are inherited from [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),  [Open-X](https://robotics-transformer-x.github.io/), [openvla-oft](https://github.com/moojink/openvla-oft), [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1).

## 📝 Citation
```
@misc{unifolm-vla-0,
  author       = {Unitree},
  title        = {UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family},
  year         = {2026},
}
```




