# Coffee Plantation Segmentation with Multi-Modal Satellite Imagery

Deep learning-based semantic segmentation of coffee plantations and eucalyptus in the Brazilian Cerrado using **Planet NICFI** optical and **Sentinel-1 SAR** time series.

> **Paper:** *Exploiting Convolutional and Transformer Networks for Coffee Plantation Mapping in Brazil Using Polarimetric-Spectral-Temporal Data from Sentinel-1 and Planet NICFI Time Series*
>
> Carvalho, O.L.F.; Carvalho Junior, O.A.; Albuquerque, A.O.; Castro Filho, H.C.; Moura, J.M.; Antony, D.S.; Silva, D.G.
>
> University of Brasilia, Brazil
>
> *Paper under review.*

---

## Overview

This repository provides the full training and evaluation pipeline for coffee plantation mapping using multi-modal satellite imagery. We evaluate **18 model configurations** (6 architectures × 3 encoders) across **4 input modalities**, using a dataset of **2,800 densely annotated tiles** (512 × 512 pixels) from Patrocinio, Minas Gerais — Brazil's largest coffee-producing region.

> **Updates (post-paper):**
> - Mask2Former (Cheng et al., 2022) is now available as a 7th architecture. See [Mask2Former](#mask2former-extension) below.
> - Temporal ablation study (3 / 6 / 9 / 12 months) using SegFormer + EfficientNet-B7 on Planet. See [`TEMPORAL_ABLATION.md`](TEMPORAL_ABLATION.md).
>
> Results in the table above reflect the original 6-architecture benchmark.

### Input Modalities

| Dataset | Bands | Description |
|---------|-------|-------------|
| **Planet** | 48 | Planet NICFI optical (B, G, R, NIR × 12 months) |
| **VH** | 30 | Sentinel-1 VH polarization (30 temporal acquisitions) |
| **VV** | 30 | Sentinel-1 VV polarization (30 temporal acquisitions) |
| **Combined** | 108 | Planet + VV + VH concatenated |

### Classes

| Class | Label |
|-------|-------|
| Background | 0 |
| Coffee | 1 |
| Eucalyptus | 2 |

---

## Results

Best model per dataset (mIoU on test set):

| Dataset | Architecture | Encoder | IoU Bkg | IoU Coffee | IoU Eucalyptus | mIoU |
|---------|-------------|---------|---------|------------|----------------|------|
| **Combined** | Segformer | EfficientNet-B7 | 95.83% | 80.63% | 83.09% | **86.52%** |
| **Planet** | Segformer | EfficientNet-B7 | 95.25% | 80.33% | 82.77% | 86.12% |
| **VH** | Segformer | EfficientNet-B7 | 89.66% | 54.27% | 53.71% | 65.88% |
| **VV** | DeepLabV3+ | EfficientNet-B7 | 90.48% | 53.25% | 56.67% | 66.80% |

Full results for all 72 model configurations are available in the paper.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cafe-segmentation.git
cd cafe-segmentation
pip install -r requirements.txt
```

> The repository includes a modified version of [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) with Segformer support. No separate installation is needed.

---

## Dataset

### Structure

Place your data in a `DATASET/` folder with the following structure:

```
DATASET/
├── Planet_GT6/
│   ├── image_train/       # .tiff files (48 bands)
│   ├── image_val/
│   ├── image_test/
│   ├── class_train/       # .png mask files
│   ├── class_val/
│   └── class_test/
├── VH_GT6/
│   ├── image_train/       # .tiff files (30 bands)
│   ├── image_val/
│   └── image_test/
├── VV_GT6/
│   ├── image_train/       # .tiff files (30 bands)
│   ├── image_val/
│   └── image_test/
└── Combined_GT6/          # Generated with prepare_combined.py
    ├── image_train/       # .tiff files (108 bands)
    ├── image_val/
    └── image_test/
```

Masks are shared across all datasets and stored in `Planet_GT6/class_*/`.

### Obtaining the Data

- **Sentinel-1 (VH, VV) and ground truth masks** are available on Hugging Face: [osmarluiz/coffee-segmentation-dataset](https://huggingface.co/datasets/osmarluiz/coffee-segmentation-dataset)
- **Planet NICFI imagery** is subject to the [NICFI license](https://www.planet.com/nicfi/). Users must obtain Planet data independently through the NICFI program archive. See `prepare_combined.py` for instructions on generating the Combined dataset.

### Preparing the Combined Dataset

If you have Planet imagery, generate the Combined dataset:

```bash
# VH + VV only (60 bands):
python prepare_combined.py --vh DATASET/VH_GT6 --vv DATASET/VV_GT6 --output DATASET/Combined_GT6

# Planet + VV + VH (108 bands, reproduces paper results):
python prepare_combined.py --planet DATASET/Planet_GT6 --vh DATASET/VH_GT6 --vv DATASET/VV_GT6 --output DATASET/Combined_GT6
```

---

## Usage

### Training

```bash
# Train a single model:
python train.py --dataset VH --model Segformer --encoder efficientnet-b7

# Train with a different loss function:
python train.py --dataset Combined --model DeepLabV3Plus --encoder resnet101 --loss focal

# Train all models for one dataset:
python train.py --dataset Planet --all-models

# Train only the 4 best models (one per dataset):
python train.py --best-only

# Train all 72 paper combinations:
python train.py --all

# Resume interrupted training:
python train.py --dataset VH --model Segformer --encoder efficientnet-b7 --resume
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | — | `Combined`, `Planet`, `VH`, or `VV` |
| `--model` | — | `Unet`, `Linknet`, `UnetPlusPlus`, `PSPNet`, `DeepLabV3Plus`, `Segformer` |
| `--encoder` | `efficientnet-b7` | `efficientnet-b7`, `resnet101`, `resnext101_32x8d` |
| `--loss` | `ce` | `ce` (CrossEntropy), `dice`, `focal` |
| `--epochs` | `100` | Number of epochs |
| `--batch-size` | `8` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--data-dir` | `DATASET` | Path to dataset root |
| `--output-dir` | `models` | Path to save models |
| `--amp` | `auto` | Autocast dtype: `auto` (bf16 only for Mask2Former), `bfloat16`, `float16`, `none`. bf16 dispenses GradScaler. |

### Evaluation

```bash
# Evaluate a specific model:
python evaluate.py --dataset Planet --model Segformer --encoder efficientnet-b7

# Evaluate the 4 best models:
python evaluate.py --best-only

# Evaluate all models and export CSV:
python evaluate.py --all --output-csv results.csv
```

### Inference on Large Images

```bash
python predict.py \
    --image path/to/large_image.tif \
    --output predictions/result.tif \
    --weights models/Planet_GT6/Segformer_efficientnet-b7/best_model.pth \
    --model Segformer --encoder efficientnet-b7 --channels 48 \
    --window 512 --stride 64
```

---

## Project Structure

```
cafe-segmentation/
├── train.py                        # Unified training script
├── evaluate.py                     # Model evaluation
├── predict.py                      # Sliding window inference
├── prepare_combined.py             # Combine VH + VV + Planet
├── requirements.txt
├── src/
│   ├── data_utils.py               # Dataset loading and transforms
│   └── loss_functions.py           # Dice, Focal, Combined losses
└── segmentation_models_pytorch/    # Modified SMP with Segformer
```

---

## Architectures

| Architecture | Reference |
|-------------|-----------|
| U-Net | Ronneberger et al. (2015) |
| LinkNet | Chaurasia & Culurciello (2017) |
| U-Net++ | Zhou et al. (2018) |
| PSPNet | Zhao et al. (2017) |
| DeepLabV3+ | Chen et al. (2018) |
| SegFormer | Xie et al. (2021) |
| Mask2Former (post-paper extension) | Cheng et al. (2022) |

---

## Mask2Former extension

Mask2Former (Cheng et al., 2022) is integrated into the bundled SMP fork at
[`segmentation_models_pytorch/decoders/mask2former/`](segmentation_models_pytorch/decoders/mask2former/)
and exposed as `smp.Mask2Former(...)`. Unlike the six per-pixel-classification
baselines, Mask2Former is a **mask-classification** model and is trained with a
full **Hungarian set-prediction** pipeline.

### Architecture

- **Backbone**: any SMP encoder (we use `efficientnet-b7`, matching the
  baselines). Accepts arbitrary `in_channels` (30 / 48 / 108).
- **Pixel decoder**: standard `nn.TransformerEncoderLayer` self-attention on
  strides 16 and 32 (jointly self-attended) plus FPN top-down 3×3 convs that
  produce strides 8 and 4. Stride-4 features are the *mask features*.
- **Transformer decoder**: 100 learnable object queries + learnable positional
  queries; 9 layers (3 rounds × 3 scales) with **masked** cross-attention (the
  previous layer's mask prediction gates which pixels each query attends to),
  self-attention among queries, then FFN.
- **Heads**: per-query class head (`C + 1` logits, the extra one is
  "no-object") and a 3-layer MLP mask-embedding head; mask logits via einsum
  between the mask embedding and the stride-4 mask features.
- **Semantic output**: `softmax(class)[:,:,:-1] ⊗ sigmoid(mask)` summed over
  queries (paper Eq. 5), upsampled 4× to full resolution.

### Training pipeline

- **HungarianMatcher** (`matcher.py`): bipartite matching between queries and
  GT instances (one instance per class present in the tile); cost = weighted
  sum of negative class probability + mask BCE + mask dice.
- **SetCriterion** (`criterion.py`): classification CE (matched queries get
  their GT class, unmatched get "no-object" down-weighted by `eos_coef=0.1`) +
  mask BCE + mask dice on matched queries. Applied to all 9 decoder layers
  (deep supervision). The paper's `[1, 15, 35]` class weights are carried into
  the per-query classification CE, so the same class-imbalance handling as the
  baselines is preserved.
- **Optimizer**: `AdamW(lr=1e-4, weight_decay=0.05)` with `clip_grad_norm=0.01`
  — Mask2Former needs a 10× smaller LR than the baselines' `Adam(lr=1e-3)`,
  otherwise the first optimizer step collapses the model into the uniform
  prediction.
- **Mixed precision**: bfloat16 autocast in the training loop (no GradScaler —
  bf16 has the fp32 exponent range). Enabled automatically for Mask2Former.
- **Validation BatchNorm**: when training from scratch with small batches the
  encoder's BN running stats take many epochs to stabilise, so
  `Mask2Former.eval()` keeps BN layers in train mode (batch statistics) to
  give a reliable validation metric.

### Deviations from the original paper (intentional)

1. **Pixel decoder without MSDeformAttn.** The original uses Multi-Scale
   Deformable Attention, which requires a CUDA extension that is painful to
   build on Windows. The pure-PyTorch self-attention + FPN substitute keeps
   memory bounded and the build dependency-free.
2. **Trained from scratch** (`encoder_weights=None`), consistent with the
   other six architectures in this repository. The original Mask2Former
   always uses an ImageNet-pretrained backbone.

### Training Mask2Former

```bash
# --amp=auto turns on bfloat16 autocast for Mask2Former automatically:
python train.py --dataset Combined --model Mask2Former --encoder efficientnet-b7

# Run all four modalities sequentially:
bash run_m2f_queue.sh
```

### Results

Mask2Former + EfficientNet-B7, test-set mIoU (trained from scratch, 100 epochs):

| Dataset | IoU Bkg | IoU Coffee | IoU Eucalyptus | mIoU | fwIoU |
|---------|---------|------------|----------------|------|-------|
| Combined | 97.11% | 86.64% | 66.68% | 83.47% | 95.29% |
| Planet | 97.13% | 86.86% | 66.61% | 83.54% | 95.35% |
| VH | 92.47% | 72.90% | 14.15% | 59.84% | 88.96% |
| VV | 92.82% | 72.83% | 16.75% | 60.80% | 89.25% |

### Smoke test

```bash
python smoke_test_mask2former.py
```

Instantiates `smp.Mask2Former(encoder='efficientnet-b7', in_channels=108,
classes=3)`, checks the dict output shapes, runs the Hungarian matcher +
SetCriterion, and verifies that the loss decreases over a few optimisation
steps on a single batch.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{carvalho2025coffee,
  title={Exploiting Convolutional and Transformer Networks for Coffee Plantation
         Mapping in Brazil Using Polarimetric-Spectral-Temporal Data from
         Sentinel-1 and Planet NICFI Time Series},
  author={Carvalho, Osmar Luiz Ferreira de and Carvalho Junior, Osmar Abilio de
          and Albuquerque, Anesmar Olino de and Castro Filho, Hugo Cristomo de
          and Moura, Joelma Mendes de and Antony, Dora Silva
          and Silva, Daniel Guerreiro e},
  year={2025},
  note={Under review}
}
```

---

## License

This code is released under the [MIT License](LICENSE).

The Sentinel-1 data and ground truth annotations are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Planet NICFI imagery is subject to the [NICFI participant license](https://www.planet.com/nicfi/).
