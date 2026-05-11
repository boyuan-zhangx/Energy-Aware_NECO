#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m src.evaluation.paper_figures --config configs/default.yaml "$@"
