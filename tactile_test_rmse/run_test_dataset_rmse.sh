#!/usr/bin/env bash
# Run all requested checkpoints serially.  Results go to ./results/*.json.
set -Eeuo pipefail

ROOT=/workspace/project/VLA-Human-Humanoid-interaction
EVAL_DIR=/workspace/project/tactile_test_rmse
DATASET=/root/tactile_ws/datasets/test-dataset/test-dataset
RESULTS="$EVAL_DIR/results/$(date +%Y%m%d_%H%M%S)"
PY="$EVAL_DIR/offline_trajectory_rmse.py"
RLDX_PY=/data/project/rldx-eval-venv/bin/python
# GPU-03's internal address is reachable directly from 02.  It is used
# instead of a local SSH alias, since aliases are not shared between hosts.
GPU03=root@gpu-server-b
SSH_OPTS=(
  -i /root/.ssh/id_ed25519
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)
UNIFOLM_PORT=18777
MAX_EPISODES=${MAX_EPISODES:-0}  # e.g. MAX_EPISODES=2 for a smoke run

mkdir -p "$RESULTS"
test -d "$DATASET"
test -x "$PY"
test -x "$RLDX_PY"
test -d /root/tactile_ws/checkpoints/Being-H/0030000
test -d /root/tactile_ws/checkpoints/extended-sep-24-with-interact-60/checkpoint-30000
test -d /root/tactile_ws/checkpoints/extended-sep-24-with-none/checkpoint-30000
test -d /data/project/chpts_rldx/checkpoint-35000
docker inspect tactile-train >/dev/null
ssh "${SSH_OPTS[@]}" "$GPU03" '
  test -f /workspace/project/uniformvla_chpts/extended_sep24_interact60_40k/checkpoints/steps_30000_pytorch_model.pt
  test -f /workspace/project/uniformvla_chpts/extended_sep24_interact60_40k/checkpoints/steps_40000_pytorch_model.pt
  test -x /workspace/project/uniformvla_work/.venv-tfds/bin/python
  test -f /workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla/deployment/model_server/run_real_eval_server.py
'

run_rldx() {
  local name=$1 checkpoint=$2 horizon=$3
  (
    cd "$ROOT/rldx"
    RLDX_ATTN_IMPL=sdpa PYTHONPATH="$ROOT/rldx" "$RLDX_PY" "$PY" \
      --backend rldx --checkpoint "$checkpoint" --dataset "$DATASET" \
      --horizon "$horizon" --max-episodes "$MAX_EPISODES" \
      --output "$RESULTS/$name.json"
  ) 2>&1 | tee "$RESULTS/$name.log"
}

run_beingh() {
  local name=$1 checkpoint=$2
  (
    cd "$ROOT/Being-H/Being-H05"
    source /data/project/beingh-venv/bin/activate
    PYTHONPATH="$ROOT/rldx:$ROOT/Being-H/Being-H05" python "$PY" \
      --backend beingh --checkpoint "$checkpoint" --dataset "$DATASET" \
      --horizon 40 --max-episodes "$MAX_EPISODES" \
      --output "$RESULTS/$name.json"
  ) 2>&1 | tee "$RESULTS/$name.log"
}

run_groot() {
  local name=$1 checkpoint=$2 horizon=$3
  local container_checkpoint=${checkpoint/#\/root\/tactile_ws/\/workspace}
  local container_dataset=/workspace/datasets/test-dataset/test-dataset
  local container_eval=/tmp/offline_trajectory_rmse.py
  local container_result=/tmp/"$name".json
  docker cp "$PY" tactile-train:"$container_eval"
  docker exec tactile-train bash -lc "
    set -euo pipefail
    source /opt/Isaac-GR00T/.venv/bin/activate
    cd /opt/Isaac-GR00T
    python $container_eval --backend groot --checkpoint $container_checkpoint \\
      --dataset $container_dataset --horizon $horizon --max-episodes $MAX_EPISODES \\
      --output $container_result
  " 2>&1 | tee "$RESULTS/$name.log"
  docker cp tactile-train:"$container_result" "$RESULTS/$name.json"
}

start_unifolm_tunnel() {
  # SSH tunnel is preferable to a persistent socat process.  The remote
  # server is started in a named tmux session and killed by cleanup().
  local checkpoint=$1
  ssh "${SSH_OPTS[@]}" "$GPU03" "tmux kill-session -t tactile-unifolm-rmse 2>/dev/null || true; \
    tmux new-session -d -s tactile-unifolm-rmse 'cd $ROOT/unifolm-vla && \
    source /workspace/project/uniformvla_work/.venv-tfds/bin/activate && \
    python deployment/model_server/run_real_eval_server.py \
      --ckpt_path $checkpoint \
      --vlm_pretrained_path /workspace/project/uniformvla_work/Unifolm-VLM-Base \
      --unnorm_key g1_sonic_brainco6dof --host 127.0.0.1 --port 8777 \
      2>&1 | tee /tmp/tactile-unifolm-rmse.log'"
  ssh "${SSH_OPTS[@]}" -N -o ExitOnForwardFailure=yes -L "$UNIFOLM_PORT:127.0.0.1:8777" "$GPU03" &
  UNIFOLM_TUNNEL_PID=$!
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$UNIFOLM_PORT/openapi.json" >/dev/null 2>&1; then return; fi
    sleep 2
  done
  echo "UnifoLM server did not become ready; see: ssh $GPU03 tmux capture-pane -pt tactile-unifolm-rmse" >&2
  ssh "${SSH_OPTS[@]}" "$GPU03" 'tail -100 /tmp/tactile-unifolm-rmse.log' >&2 || true
  return 1
}

stop_unifolm() {
  if [[ -n "${UNIFOLM_TUNNEL_PID:-}" ]]; then
    kill "$UNIFOLM_TUNNEL_PID" 2>/dev/null || true
    wait "$UNIFOLM_TUNNEL_PID" 2>/dev/null || true
    unset UNIFOLM_TUNNEL_PID
  fi
  ssh "${SSH_OPTS[@]}" "$GPU03" 'tmux kill-session -t tactile-unifolm-rmse 2>/dev/null || true' || true
}

run_unifolm() {
  local name=$1 checkpoint=$2
  local eval_status=0
  start_unifolm_tunnel "$checkpoint"
  set +e
  (
    "$RLDX_PY" "$PY" \
      --backend unifolm --checkpoint "$checkpoint" --dataset "$DATASET" \
      --horizon 40 --unifolm-url "http://127.0.0.1:$UNIFOLM_PORT/act" \
      --max-episodes "$MAX_EPISODES" --output "$RESULTS/$name.json"
  ) 2>&1 | tee "$RESULTS/$name.log"
  eval_status=${PIPESTATUS[0]}
  set -e
  ssh "${SSH_OPTS[@]}" "$GPU03" 'cat /tmp/tactile-unifolm-rmse.log' \
    > "$RESULTS/$name.server.log" 2>&1 || true
  stop_unifolm
  return "$eval_status"
}

cleanup() { stop_unifolm; }
trap cleanup EXIT INT TERM

run_beingh beingh_0030000 /root/tactile_ws/checkpoints/Being-H/0030000
run_groot groot_interact60_30000 /root/tactile_ws/checkpoints/extended-sep-24-with-interact-60/checkpoint-30000 50
run_groot groot_none_30000 /root/tactile_ws/checkpoints/extended-sep-24-with-none/checkpoint-30000 50
run_rldx rldx_35000 /data/project/chpts_rldx/checkpoint-35000 40

# These are the two checkpoints in the directory supplied for GPU-03.
run_unifolm unifolm_30000 /workspace/project/uniformvla_chpts/extended_sep24_interact60_40k/checkpoints/steps_30000_pytorch_model.pt
run_unifolm unifolm_40000 /workspace/project/uniformvla_chpts/extended_sep24_interact60_40k/checkpoints/steps_40000_pytorch_model.pt

echo "Finished. JSON reports and logs: $RESULTS"
