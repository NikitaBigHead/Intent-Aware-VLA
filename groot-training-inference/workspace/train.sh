docker exec -it tactile-train bash


source /opt/Isaac-GR00T/.venv/bin/activate
cd /opt/Isaac-GR00T


MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/extended-sep-24-with-none \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-24-with-none  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log


MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/interact_percents/extended-sep-24-with-interact-30 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-24-with-interact-30  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log



MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/interact_percents/extended-sep-24-with-interact-60 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-24-with-interact-60  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log


MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/interact_percents/extended-sep-24-with-interact-90 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-24-with-interact-90  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log





MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/interact_percents/extended-sep-24-with-interact-90 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-24-with-interact-90  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log





MAX_STEPS=50000 SAVE_STEPS=5000 GLOBAL_BATCH_SIZE=32 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/extended-sep-29-with-none-cleaned \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-29-with-none-cleaned  \
  -- --save_total_limit 3 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log



MAX_STEPS=30000 SAVE_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/\
  --dataset-path /workspace/datasets/interact_percents/extended-sep-29-with-interact-60 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/affective_g1_config_two_cam.py \
  --output-dir /workspace/checkpoints/extended-sep-29-with-none-interact-60  \
  -- --save_total_limit 1 \
  2>&1 | tee /workspace/logs/train_$(date +%Y%m%d_%H%M%S).log

