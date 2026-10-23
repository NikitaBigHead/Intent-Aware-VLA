#!/usr/bin/env bash
# Train + evaluate all encoder x temporal combinations on the SAME split, T and resolution.
# Usage: bash scripts/run_grid.sh [fold] [seed]     (extra overrides via EXTRA="a.b=c ...")
set -euo pipefail
FOLD=${1:-0}
SEED=${2:-0}
EXTRA=${EXTRA:-}
CONFIG=${CONFIG:-configs/base.yaml}
for ENC in tinyvit mobilevit efficientformer; do
  for TMP in vtn videolightformer; do
    NAME="${ENC}_${TMP}_f${FOLD}_s${SEED}"
    echo "=== ${NAME}"
    python scripts/train.py --config "${CONFIG}" --set \
      model.encoder.name=${ENC} model.temporal.name=${TMP} \
      split.fold=${FOLD} experiment.seed=${SEED} experiment.name=${NAME} ${EXTRA}
  done
done
python scripts/summarize.py "${OUT_DIR:-runs}"
