# Energy-Aware NECO

Single-pass pixel-wise out-of-distribution (OOD) detection for semantic segmentation through hybrid feature geometry and logit energy scoring.

Accepted at the LoWi Workshop @ ICRA 2026.

---

## Overview

Reliable dense perception is critical for autonomous driving and mobile robotics.  
While uncertainty-based methods such as Deep Ensemble and Monte Carlo Dropout provide strong OOD detection performance, they require repeated stochastic inference and are expensive for real-time edge deployment.

This project proposes **Energy-Aware NECO**, a lightweight single-pass OOD detector that combines:

- **NECO-inspired feature geometry**
- **Energy-based logit evidence**

under a unified post-hoc scoring framework.

The method reuses decoder features and logits already produced by the segmentation network, avoiding repeated inference while remaining competitive with stronger uncertainty baselines.

---

## Highlights

- Single-pass dense OOD detection
- Hybrid geometry + energy scoring
- No OOD data required during fitting
- Competitive with ensemble uncertainty baselines
- Designed for edge-oriented robotics deployment
- Evaluated on miniMUAD with true pixel-level OOD masks

---

## Method

The proposed detector combines two complementary signals:

### 1. Geometry Branch (NECO-style)

A centered projection ratio measures how strongly decoder features align with the learned in-distribution principal subspace.

### 2. Energy Branch

A logit-based Energy score captures abnormal activation evidence that geometry alone may miss.

### 3. Hybrid Fusion

Both scores are standardized using an ID-only validation split and fused into a final OOD score.

---

## Pipeline

<p align="center">
  <img src="assets/pipeline.png" width="90%">
</p>

---

## Main Quantitative Result

| Method | AUROC |
|---|---|
| Ensemble Predictive Entropy | 0.8124 |
| NECO-only | 0.8280 |
| Energy-only | 0.8171 |
| Hybrid (NECO + Energy) | **0.8539** |

The hybrid detector achieves the best global ranking performance on miniMUAD under the same pixel-wise OOD evaluation protocol.

---

## Qualitative OOD Maps

<p align="center">
  <img src="assets/qualitative.png" width="95%">
</p>

The hybrid detector produces cleaner and more spatially continuous OOD responses than ensemble predictive entropy.

---

## Operating-Point Trade-off

<p align="center">
  <img src="assets/roc.png" width="70%">
</p>

The hybrid method dominates over a broad region of the ROC curve, while ensemble uncertainty remains advantageous in the extreme high-recall regime.

---

## Repository Structure

```text
Energy-Aware_NECO/
├── configs/                # Experiment configurations
├── datasets/               # Dataset cache and notes
├── checkpoints/            # Trained checkpoints
├── scripts/                # Training and evaluation scripts
├── src/
│   ├── evaluation/         # Benchmark and paper evaluation
│   ├── models/             # Segmentation models
│   ├── scoring/            # NECO, Energy, uncertainty scoring
│   └── utils/              # Utilities
├── results/                # Generated metrics and figures
├── paper/                  # Paper PDF
├── poster/                 # Poster PDF
├── assets/                 # README figures
├── environment.yml
├── requirements.txt
└── README.md

Installation

Create the environment using Conda:

conda env create -f environment.yml
conda activate neco-energy

or using pip:

pip install -r requirements.txt
Usage
Training
bash scripts/train.sh
Evaluation
bash scripts/eval.sh
Benchmark
bash scripts/benchmark.sh
Regenerate Paper Figures
bash scripts/paper_figures.sh
Reproducibility

Run quick verification checks before long GPU runs:

python -m compileall src

python -m src.evaluation.train --help
python -m src.evaluation.eval --help
python -m src.evaluation.benchmark --help
python -m src.evaluation.paper_figures --help
Paper

PDF version:

paper/Energy_Aware_NECO_ICRA2026.pdf
Poster

ICRA 2026 LoWi poster:

poster/LOWI_ICRA2026_Poster.pdf
Citation
@misc{zhang2026energyaware,
  title={Energy-Aware NECO for Single-Pass Pixel-wise Out-of-Distribution Detection in Semantic Segmentation},
  author={Boyuan Zhang and Huanshan Huang and Yifei Cao},
  year={2026},
  note={Accepted at LoWi Workshop, ICRA 2026}
}
Acknowledgement

This work received a LoWi 2026 travel grant sponsored by TIANBOT.

Contact

Boyuan Zhang
École Polytechnique, Institut Polytechnique de Paris

LinkedIn:
https://www.linkedin.com/in/boyuan-zhang-493776216
