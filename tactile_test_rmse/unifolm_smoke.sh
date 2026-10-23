#!/usr/bin/env bash
set -Eeuo pipefail

step=${1:?checkpoint step, e.g. 30000}
eval_dir=/workspace/project/tactile_test_rmse
checkpoint=/workspace/project/uniformvla_chpts/extended_sep24_interact60_40k/checkpoints/steps_${step}_pytorch_model.pt
max_episodes=${MAX_EPISODES:-1}
output=${OUTPUT:-$eval_dir/smoke/unifolm_${step}.json}
gpu03=root@gpu-server-b
ssh_opts=(-i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
port=18777

cleanup() {
  if [[ -n "${tunnel_pid:-}" ]]; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
  ssh "${ssh_opts[@]}" "$gpu03" 'tmux kill-session -t tactile-unifolm-rmse 2>/dev/null || true' || true
}
trap cleanup EXIT INT TERM

ssh "${ssh_opts[@]}" "$gpu03" "tmux kill-session -t tactile-unifolm-rmse 2>/dev/null || true; \
  tmux new-session -d -s tactile-unifolm-rmse 'cd /workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla && \
  source /workspace/project/uniformvla_work/.venv-tfds/bin/activate && \
  python deployment/model_server/run_real_eval_server.py \
    --ckpt_path $checkpoint \
    --vlm_pretrained_path /workspace/project/uniformvla_work/Unifolm-VLM-Base \
    --unnorm_key g1_sonic_brainco6dof --host 127.0.0.1 --port 8777 \
    > /tmp/tactile-unifolm-rmse.log 2>&1'"

ssh "${ssh_opts[@]}" -N -o ExitOnForwardFailure=yes -L "$port:127.0.0.1:8777" "$gpu03" &
tunnel_pid=$!
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$port/openapi.json" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$port/openapi.json" >/dev/null

mkdir -p "$(dirname "$output")"
/data/project/rldx-eval-venv/bin/python "$eval_dir/offline_trajectory_rmse.py" \
  --backend unifolm --checkpoint "$checkpoint" \
  --dataset /root/tactile_ws/datasets/test-dataset/test-dataset \
  --horizon 40 --unifolm-url "http://127.0.0.1:$port/act" \
  --max-episodes "$max_episodes" --output "$output"
