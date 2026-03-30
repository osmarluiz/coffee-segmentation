#!/usr/bin/env python
"""
Evaluate trained segmentation models on the test set.

Usage examples:
    # Evaluate a single model:
    python evaluate.py --dataset Planet --model Segformer --encoder efficientnet-b7

    # Evaluate all trained models found in the models directory:
    python evaluate.py --all

    # Evaluate the 4 best models:
    python evaluate.py --best-only
"""

import argparse
import os
import sys
import json

import torch
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp
from src.data_utils import CustomDataset


# ─── Constants ───────────────────────────────────────────────────────────────

DATASETS = {
    'Combined': {'channels': 108, 'dir': 'Combined_GT6'},
    'Planet':   {'channels': 48,  'dir': 'Planet_GT6'},
    'VH':       {'channels': 30,  'dir': 'VH_GT6'},
    'VV':       {'channels': 30,  'dir': 'VV_GT6'},
}

MODELS = ['Unet', 'Linknet', 'UnetPlusPlus', 'PSPNet', 'DeepLabV3Plus', 'Segformer']
ENCODERS = ['efficientnet-b7', 'resnet101', 'resnext101_32x8d']
NUM_CLASSES = 3

BEST_MODELS = [
    ('Combined', 'Segformer',     'efficientnet-b7'),
    ('Planet',   'Segformer',     'efficientnet-b7'),
    ('VH',       'Segformer',     'efficientnet-b7'),
    ('VV',       'DeepLabV3Plus', 'efficientnet-b7'),
]


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate segmentation models')

    p.add_argument('--dataset', type=str, choices=list(DATASETS.keys()))
    p.add_argument('--model', type=str, choices=MODELS)
    p.add_argument('--encoder', type=str, choices=ENCODERS, default='efficientnet-b7')

    p.add_argument('--all', action='store_true', help='Evaluate all models in --models-dir')
    p.add_argument('--best-only', action='store_true', help='Evaluate only the 4 best models')

    p.add_argument('--data-dir', type=str, default='DATASET')
    p.add_argument('--models-dir', type=str, default='models')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--output-csv', type=str, default=None,
                   help='Save results to CSV file')

    return p.parse_args()


def load_test_data(data_dir, dataset_name, num_workers=0):
    """Load test set for a given dataset."""
    ds_dir = os.path.join(data_dir, DATASETS[dataset_name]['dir'])
    mask_dir = os.path.join(data_dir, 'Planet_GT6')

    test_imgs  = sorted(glob(os.path.join(ds_dir, 'image_test/*.tiff')))
    test_masks = sorted(glob(os.path.join(mask_dir, 'class_test/*.png')))

    assert len(test_imgs) > 0, f"No test images in {ds_dir}/image_test/"

    val_tf = transforms.ToTensor()
    test_ds = CustomDataset(test_imgs, test_masks, transform=val_tf, transform_label=None)
    return DataLoader(test_ds, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def evaluate_model(model, data_loader, device):
    """Compute per-class IoU, mIoU, and fwIoU."""
    model.eval()
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(dim=1)
            conf_matrix += confusion_matrix(
                masks.view(-1).cpu().numpy(),
                preds.view(-1).cpu().numpy(),
                labels=range(NUM_CLASSES))

    ious = []
    for i in range(NUM_CLASSES):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ious.append(iou)

    miou = np.mean(ious)
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    fwiou = np.sum(freq * ious)

    return {
        'ious': ious,
        'mIoU': miou,
        'fwIoU': fwiou,
        'confusion_matrix': conf_matrix.tolist(),
    }


def evaluate_single(dataset, model_name, encoder, models_dir, data_dir, device):
    """Load and evaluate a single model."""
    in_channels = DATASETS[dataset]['channels']

    model_dir = os.path.join(models_dir, DATASETS[dataset]['dir'],
                             f"{model_name}_{encoder}")
    model_path = os.path.join(model_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None

    model = getattr(smp, model_name)(
        encoder_name=encoder,
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation='softmax',
        in_channels=in_channels,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    test_loader = load_test_data(data_dir, dataset)
    results = evaluate_model(model, test_loader, device)

    return results


def main():
    args = parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # Determine what to evaluate
    if args.best_only:
        combinations = BEST_MODELS
    elif args.all:
        combinations = [
            (ds, model, enc)
            for ds in DATASETS
            for model in MODELS
            for encoder in ENCODERS
            for enc in [encoder]
        ]
    elif args.dataset and args.model:
        combinations = [(args.dataset, args.model, args.encoder)]
    else:
        print("Error: specify --dataset + --model, or use --all / --best-only")
        sys.exit(1)

    all_results = []

    for ds, model_name, encoder in combinations:
        print(f"\nEvaluating {model_name}+{encoder} on {ds}...")
        result = evaluate_single(ds, model_name, encoder,
                                 args.models_dir, args.data_dir, args.device)
        if result is None:
            continue

        all_results.append({
            'dataset': ds,
            'model': model_name,
            'encoder': encoder,
            'IoU_0': result['ious'][0] * 100,
            'IoU_1': result['ious'][1] * 100,
            'IoU_2': result['ious'][2] * 100,
            'mIoU': result['mIoU'] * 100,
            'fwIoU': result['fwIoU'] * 100,
        })

        print(f"  IoU: {result['ious'][0]*100:.2f}% | {result['ious'][1]*100:.2f}% | {result['ious'][2]*100:.2f}%")
        print(f"  mIoU: {result['mIoU']*100:.2f}%  fwIoU: {result['fwIoU']*100:.2f}%")

    # Results table
    if all_results:
        print(f"\n{'='*85}")
        print(f"{'Dataset':<12} {'Model':<18} {'Encoder':<22} {'IoU_0':>7} {'IoU_1':>7} {'IoU_2':>7} {'mIoU':>7}")
        print('-' * 85)
        for r in all_results:
            print(f"{r['dataset']:<12} {r['model']:<18} {r['encoder']:<22} "
                  f"{r['IoU_0']:>6.2f}% {r['IoU_1']:>6.2f}% {r['IoU_2']:>6.2f}% "
                  f"{r['mIoU']:>6.2f}%")

    # Save to CSV
    if args.output_csv and all_results:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == '__main__':
    main()
