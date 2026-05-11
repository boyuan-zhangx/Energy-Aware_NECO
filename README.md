# Energy-Aware NECO

Single-pass pixel-wise OOD detection for semantic segmentation with a hybrid NECO and energy score.

## Structure

- `configs/`: experiment configuration.
- `datasets/`: dataset cache location and dataset notes.
- `checkpoints/`: trained model and detector states.
- `src/models/`: U-Net implementation.
- `src/scoring/`: uncertainty, energy, and NECO scoring methods.
- `src/evaluation/`: training, evaluation, and benchmark code.
- `src/utils/`: config, data, device, and visualization helpers.
- `scripts/`: standalone train, evaluation, and benchmark commands.
- `results/`: metrics, histories, and generated outputs.

## Setup

```bash
conda env create -f environment.yml
conda activate neco-energy
```

or:

```bash
pip install -r requirements.txt
```

## Usage

```bash
bash scripts/train.sh
bash scripts/eval.sh
bash scripts/benchmark.sh
bash scripts/paper_figures.sh
```

Each script accepts extra arguments and forwards them to the Python module. For example:

```bash
bash scripts/train.sh --epochs 5
FORCE_TRAIN=1 bash scripts/train.sh --epochs 5
bash scripts/benchmark.sh --max-batches 10 --batch-size 1 --num-workers 0
bash scripts/benchmark.sh --methods all --skip-runtime
bash scripts/paper_figures.sh --batch-size 1 --num-workers 0
```

`train.sh` follows the notebook's default `skip_training=True` behavior: when
`checkpoints/unet.pth` already exists it skips retraining and regenerates the
training-curve plot from `results/training_history.json`. Set `FORCE_TRAIN=1`
to intentionally train and overwrite the default checkpoint.

`benchmark.sh` defaults to the paper protocol: Deep Ensemble predictive entropy plus NECO-only, Energy-only, and Hybrid NECO+Energy under the same pixel-wise ID/OOD protocol. `--methods all` adds the non-training exploratory uncertainty baselines used during development.

`paper_figures.sh` regenerates the notebook/paper-style outputs in `results/paper_figures/`: calibration figures, the single-scene OOD qualitative figure, multi-scene OOD maps, score distributions, feature-geometry diagnostic, ROC tail view, condition-wise bars, and `paper_results.json`.

## Verification

Run syntax and import checks before launching long GPU jobs:

```bash
python -m compileall src
python -m src.evaluation.train --help
python -m src.evaluation.eval --help
python -m src.evaluation.benchmark --help
python -m src.evaluation.paper_figures --help
```
