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
