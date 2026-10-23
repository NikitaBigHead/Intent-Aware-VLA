# Cross-model trajectory RMSE evaluation

Project directory: [`../tactile_test_rmse/`](../tactile_test_rmse/)

This source-only snapshot came from server 2,
`/workspace/project/tactile_test_rmse`. Generated reports, logs, smoke-test output,
and Python caches are excluded.

The evaluator
[`offline_trajectory_rmse.py`](../tactile_test_rmse/offline_trajectory_rmse.py)
supports four backends: `groot`, `rldx`, `beingh`, and `unifolm`.

Basic interface:

```bash
python offline_trajectory_rmse.py \
  --backend <groot|rldx|beingh|unifolm> \
  --checkpoint /path/to/checkpoint \
  --dataset /path/to/test-dataset \
  --horizon <40-or-50> \
  --max-episodes 0 \
  --output /path/to/report.json
```

`--max-episodes 0` evaluates all episodes. UnifoLM uses an HTTP model server;
override `--unifolm-url` if it is not available at
`http://127.0.0.1:8777/act`.

[`run_test_dataset_rmse.sh`](../tactile_test_rmse/run_test_dataset_rmse.sh)
is the original orchestration script for all four model families. It contains
server-specific paths, container names, and SSH settings and therefore serves
as a reproducibility record; update these values before running elsewhere.
[`unifolm_smoke.sh`](../tactile_test_rmse/unifolm_smoke.sh) runs a small
UnifoLM check, and
[`run_remaining_unifolm.sh`](../tactile_test_rmse/run_remaining_unifolm.sh)
runs the remaining full UnifoLM evaluations.

The evaluation requires the external test dataset, model checkpoints, and the
environment for the selected backend. None of these large artifacts are
included in the source submission.
