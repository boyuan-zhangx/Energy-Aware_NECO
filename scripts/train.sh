#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  python -m src.evaluation.train --config configs/default.yaml "$@"
else
  python -m src.evaluation.train --config configs/default.yaml --skip-if-checkpoint-exists "$@"
fi
