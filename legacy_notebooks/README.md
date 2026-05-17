# Legacy notebooks

Exploratory Jupyter notebooks from earlier experiment rounds, kept for
reference / provenance. **These are not the canonical pipeline** — the
canonical training/eval code lives in the repository root
(`train.py`, `evaluate.py`, `predict.py`, `temporal_ablation.py`).

They are committed so the team can cross-check which configuration produced
which numbers.

## `COFFEE-FINALv2.ipynb`

A 4-model × 4-modality run. **Its configuration differs from both
`COFFEE-FINAL` and the repo scripts** — check carefully before comparing
numbers against it:

| Aspect | COFFEE-FINALv2 | Repo scripts (`train.py`) |
|--------|----------------|---------------------------|
| Encoder | `mit_b5` | `efficientnet-b7` |
| Learning rate | `1e-4` | `1e-3` (baselines) |
| Optimizer | `AdamW(weight_decay=0.01)` | `Adam` (baselines) |
| Final activation | `softmax2d` | `softmax` |
| Best-model selection | **val mIoU** | **val loss** |
| Epochs | 50 | 100 |
| Models | Segformer, DeepLabV3Plus, PSPNet, Unet | 7 architectures |

The notebook's embedded results table (Segformer + `mit_b5`: Combined
84.19% mIoU, Planet 82.29%) does **not** match the paper's published
baseline (Segformer + EfficientNet-B7, Combined 86.52%). The published
numbers come from a different run/configuration — identify that run before
treating any notebook here as the paper baseline.
