# Temporal Ablation Study

Addresses **Reviewer R2.7** (revision 2):

> *"compare 3-month, 6-month, 9-month, and 12-month temporal inputs, and
> report the resulting mIoU and class-specific IoU metrics... identify the
> most informative phenological windows for coffee mapping"*

Trains [SegFormer + EfficientNet-B7](train.py) on Planet NICFI subsets, selecting
different consecutive month windows. Each Planet tile has **48 bands = 4
spectral bands (B, G, R, NIR) × 12 monthly composites**. Bands are selected at
runtime via `rasterio.read(indexes=...)` — no disk duplication.

---

## Windows Trained

| Window | Months | Bands | Phenological coverage |
|--------|--------|-------|-----------------------|
| `3-cons-Jan` | Jan, Feb, Mar | 12 | Q1 — rainy season |
| `3-cons-Apr` | Apr, May, Jun | 12 | Q2 — start of dry season |
| `3-cons-Jul` | Jul, Aug, Sep | 12 | Q3 — **harvest** |
| `3-cons-Oct` | Oct, Nov, Dec | 12 | Q4 — flowering + early rains |
| `6-cons-1H`  | Jan–Jun | 24 | First semester |
| `6-cons-Apr` | Apr–Sep | 24 | Maturation + harvest |
| `6-cons-2H`  | Jul–Dec | 24 | Second semester |
| `9-cons`     | Jan–Sep | 36 | First 9 months |
| `9-cons-Dec` | Apr–Dec | 36 | Last 9 months |
| **`12-full`** | Jan–Dec | 48 | **Full-year baseline** |

---

## Run on PC1

After `git pull`, run the 12-month window with the same setup as the rest of
the ablation (bfloat16 AMP, batch 8, 100 epochs, Adam lr=1e-3, weighted CE
[1, 15, 35]):

```bash
# From the repository root:
python temporal_ablation.py --window 12-full --amp bfloat16
```

This expects the Planet dataset at `DATASET/Planet_GT6/` (the same path used
by `train.py`). Output goes to `models/temporal_ablation/`.

**Expected runtime on a single RTX 4090: ~12–14 hours** for 100 epochs at
~7–10 min/epoch with 48 channels. The script writes incremental checkpoints
every epoch (`last_model.pth`) and resumes from the last completed epoch if
the process crashes or is killed.

**Expected mIoU** ≈ 86 % (matches the published paper baseline of 86.10%
within ±1% margin — this run validates that bfloat16 AMP does not materially
change the result).

---

## Run all windows (PC2)

```bash
python temporal_ablation.py --all --amp bfloat16
```

---

## Selection criterion: two checkpoints saved

The script saves **two** best checkpoints per window:

- `best_model_loss.pth` — selected by `min val_loss` (matches the paper)
- `best_model_miou.pth` — selected by `max val_mIoU` (auxiliary criterion)

Both are evaluated on the test set, producing two rows per window in
`temporal_ablation_results.csv`:

| Selection | Use for |
|-----------|---------|
| `val_loss` | Direct comparison with the published baseline (paper criterion) |
| `val_miou` | Discussion of loss-metric divergence (see Section 4 below) |

---

## Output

```
models/temporal_ablation/
├── ablation.log
├── temporal_ablation_results.csv
└── Planet_Segformer_efficientnet-b7_<window>/
    ├── best_model_loss.pth
    ├── best_model_miou.pth
    ├── last_model.pth
    ├── training_status.json
    └── test_results.json
```

`temporal_ablation_results.csv` columns:

```
Window, Months, N_Months, In_Channels, Selection,
Best_Epoch, Best_Val_Loss, Best_Val_mIoU,
IoU_0, IoU_1, IoU_2, mIoU, fwIoU
```

---

## Resume after a crash

Just re-run the same command. The script:

1. Skips windows whose row is already in `temporal_ablation_results.csv`.
2. For in-progress windows, resumes from `training_status.json` and reloads
   `last_model.pth`.

On Windows, a watchdog that relaunches `temporal_ablation.py` on crash is
recommended (PyTorch 2.6 + CUDA 12.6 occasionally crashes in
`Adam.step` with `'Tensor' object is not callable` — re-launch resumes from
the last epoch without losing more than ~1 epoch of progress).

---

## Reproducibility notes

| Hyperparameter | Value | Same as paper? |
|----------------|-------|----------------|
| Architecture   | SegFormer + EfficientNet-B7 | yes |
| Encoder weights | None (from scratch) | yes |
| Loss           | Weighted CE [1, 15, 35] | yes |
| Optimizer      | Adam, lr = 1e-3 | yes |
| Epochs         | 100 | yes |
| Batch train / val / test | 8 / 32 / 32 | yes |
| Augmentation   | RandomHFlip + RandomVFlip | yes |
| Best-model selection | val_loss (paper) + val_mIoU (extra) | extended |
| AMP            | bfloat16 autocast | **NEW** (paper ran FP32) |
| `num_workers`  | 4 train / 2 val/test | **NEW** (paper used 0) |

The two "NEW" items are pure speed optimizations and have no documented
effect on optimization trajectory. The `12-full` window exists in this study
specifically to confirm this empirically against the published 86.10%
baseline.

---

## Why two selection criteria?

In windows with ≥36 input channels (`9-cons`, `9-cons-Dec`, `12-full`), we
observe **loss–metric divergence**: `val_loss` plateaus around epoch 40–60
but `val_mIoU` keeps improving through epoch 100. The paper's class weighting
`[1, 15, 35]` is mathematically insufficient to balance the loss for a
0.8% minority class (Eucalyptus); even with weight 35, Eucalyptus contributes
~9% of the weighted loss signal versus ~33% of mIoU. As a result, selecting
the best checkpoint by `val_loss` may catch the model before it has fully
learned the minority class, while `val_mIoU` recovers it. Both criteria are
saved so the discussion can be quantitative rather than speculative.
