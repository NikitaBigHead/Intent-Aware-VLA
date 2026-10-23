#!/usr/bin/env bash
# Finish only the two GPU-03 UnifoLM evaluations that were missing from the
# original full run.  The already completed Being-H/GR00T/RLDX reports remain
# untouched.
set -Eeuo pipefail

eval_dir=/workspace/project/tactile_test_rmse
results=${RESULTS_DIR:-$eval_dir/results/20261001_000659}
gpu03=root@gpu-server-b
ssh_opts=(-i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes)

mkdir -p "$results"

free_mib=$(ssh "${ssh_opts[@]}" "$gpu03" \
  "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
if (( free_mib < 20000 )); then
  echo "GPU-03 has only ${free_mib} MiB free; need at least 20000 MiB. Try later." >&2
  exit 1
fi

run_one() {
  local step=$1 name=unifolm_$1 status=0
  set +e
  MAX_EPISODES=0 OUTPUT="$results/$name.json" \
    "$eval_dir/unifolm_smoke.sh" "$step" 2>&1 | tee "$results/$name.log"
  status=${PIPESTATUS[0]}
  set -e
  ssh "${ssh_opts[@]}" "$gpu03" 'cat /tmp/tactile-unifolm-rmse.log' \
    > "$results/$name.server.log" 2>&1 || true
  return "$status"
}

run_one 30000
run_one 40000
echo "Finished missing UnifoLM evaluations: $results"
