#!/usr/bin/env python
"""
Unified training script for coffee plantation segmentation.

Usage examples:
    # Train a single model:
    python train.py --dataset VH --model Segformer --encoder efficientnet-b7

    # Train with a specific loss function:
    python train.py --dataset Combined --model DeepLabV3Plus --encoder resnet101 --loss focal

    # Train all models for one dataset:
    python train.py --dataset Planet --all-models

    # Train only the best model per dataset (4 models):
    python train.py --best-only

    # Train all paper combinations (72 models):
    python train.py --all
"""

import argparse
import os
import sys
import json
import time

import torch
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp
from src.data_utils import CustomDataset
from src.loss_functions import MulticlassDiceLoss, MulticlassFocalLoss


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
CLASS_WEIGHTS = [1.0, 15.0, 35.0]

# Best model for each dataset (from paper results)
BEST_MODELS = [
    ('Combined', 'Segformer',     'efficientnet-b7'),
    ('Planet',   'Segformer',     'efficientnet-b7'),
    ('VH',       'Segformer',     'efficientnet-b7'),
    ('VV',       'DeepLabV3Plus', 'efficientnet-b7'),
]

# All combinations tested in the paper
ALL_COMBINATIONS = [
    (ds, model, encoder)
    for ds in DATASETS
    for model in MODELS
    for encoder in ENCODERS
]


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train coffee plantation segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single model
    p.add_argument('--dataset', type=str, choices=list(DATASETS.keys()),
                   help='Dataset to train on')
    p.add_argument('--model', type=str, choices=MODELS,
                   help='Model architecture')
    p.add_argument('--encoder', type=str, choices=ENCODERS,
                   default='efficientnet-b7', help='Encoder backbone')
    p.add_argument('--loss', type=str, choices=['ce', 'dice', 'focal'],
                   default='ce', help='Loss function (default: ce)')

    # Batch training
    p.add_argument('--all', action='store_true',
                   help='Train all 72 paper combinations')
    p.add_argument('--all-models', action='store_true',
                   help='Train all models for the specified --dataset')
    p.add_argument('--best-only', action='store_true',
                   help='Train only the 4 best models (one per dataset)')

    # Hyperparameters
    p.add_argument('--epochs', type=int, default=100,
                   help='Number of training epochs (default: 100)')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Training batch size (default: 8)')
    p.add_argument('--lr', type=float, default=0.001,
                   help='Learning rate (default: 0.001)')
    p.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader workers (default: 0)')

    # Paths
    p.add_argument('--data-dir', type=str, default='DATASET',
                   help='Root directory with dataset folders (default: DATASET)')
    p.add_argument('--output-dir', type=str, default='models',
                   help='Directory to save models (default: models)')

    # Other
    p.add_argument('--resume', action='store_true',
                   help='Resume training from last checkpoint')
    p.add_argument('--device', type=str, default=None,
                   help='Device (cuda/cpu). Auto-detected if not set.')

    return p.parse_args()


# ─── Model creation ──────────────────────────────────────────────────────────

def create_model(model_name, encoder_name, in_channels):
    """Create a segmentation model using SMP."""
    return getattr(smp, model_name)(
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation='softmax',
        in_channels=in_channels,
    )


# ─── Loss function ───────────────────────────────────────────────────────────

def get_loss_fn(loss_name, device):
    """Create loss function by name."""
    weights = torch.tensor(CLASS_WEIGHTS).to(device)

    if loss_name == 'ce':
        return smp.utils.losses.CrossEntropyLoss(weight=weights)
    elif loss_name == 'dice':
        loss = MulticlassDiceLoss(num_classes=NUM_CLASSES, weight=CLASS_WEIGHTS)
        loss._name = 'dice_loss'
        return loss
    elif loss_name == 'focal':
        loss = MulticlassFocalLoss(num_classes=NUM_CLASSES, alpha=CLASS_WEIGHTS, gamma=2.0)
        loss._name = 'focal_loss'
        return loss


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir, dataset_name, batch_size, num_workers):
    """Load train/val/test data loaders."""
    ds_dir = os.path.join(data_dir, DATASETS[dataset_name]['dir'])

    train_imgs = sorted(glob(os.path.join(ds_dir, 'image_train/*.tiff')))
    val_imgs   = sorted(glob(os.path.join(ds_dir, 'image_val/*.tiff')))
    test_imgs  = sorted(glob(os.path.join(ds_dir, 'image_test/*.tiff')))

    # Masks always come from Planet_GT6
    mask_dir = os.path.join(data_dir, 'Planet_GT6')
    train_masks = sorted(glob(os.path.join(mask_dir, 'class_train/*.png')))
    val_masks   = sorted(glob(os.path.join(mask_dir, 'class_val/*.png')))
    test_masks  = sorted(glob(os.path.join(mask_dir, 'class_test/*.png')))

    assert len(train_imgs) > 0, f"No training images found in {ds_dir}/image_train/"
    assert len(train_masks) > 0, f"No training masks found in {mask_dir}/class_train/"

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    val_tf = transforms.ToTensor()

    train_ds = CustomDataset(train_imgs, train_masks, transform=train_tf, transform_label=train_tf)
    val_ds   = CustomDataset(val_imgs,   val_masks,   transform=val_tf,   transform_label=None)
    test_ds  = CustomDataset(test_imgs,  test_masks,  transform=val_tf,   transform_label=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"  Data: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    return train_loader, val_loader, test_loader


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset, returning per-class IoU, mIoU, and fwIoU."""
    model.eval()
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(dim=1)

            pred_flat = preds.view(-1).cpu().numpy()
            mask_flat = masks.view(-1).cpu().numpy()
            conf_matrix += confusion_matrix(mask_flat, pred_flat,
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

    return {'ious': ious, 'mIoU': miou, 'fwIoU': fwiou}


# ─── Training status (resume) ────────────────────────────────────────────────

def save_status(model_dir, epoch, best_loss, best_epoch):
    with open(os.path.join(model_dir, 'training_status.json'), 'w') as f:
        json.dump({
            'last_completed_epoch': epoch,
            'best_loss': best_loss,
            'best_model_epoch': best_epoch,
        }, f)


def load_status(model_dir):
    path = os.path.join(model_dir, 'training_status.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'last_completed_epoch': -1, 'best_loss': float('inf'), 'best_model_epoch': -1}


# ─── Train a single model ────────────────────────────────────────────────────

def train_single(dataset, model_name, encoder, loss_name, args):
    """Train one model configuration and evaluate on test set."""
    device = args.device
    in_channels = DATASETS[dataset]['channels']

    print(f"\n{'='*60}")
    print(f"  {model_name} + {encoder}")
    print(f"  Dataset: {dataset} ({in_channels} channels)  |  Loss: {loss_name}")
    print(f"{'='*60}")

    # Output directory
    model_dir = os.path.join(args.output_dir, DATASETS[dataset]['dir'],
                             f"{model_name}_{encoder}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pth')

    # Data
    train_loader, val_loader, test_loader = load_data(
        args.data_dir, dataset, args.batch_size, args.num_workers)

    # Model
    model = create_model(model_name, encoder, in_channels)
    model.to(device)

    # Loss and metrics
    loss_fn = get_loss_fn(loss_name, device)
    metrics = [smp.utils.metrics.mIoU()]

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    status = load_status(model_dir) if args.resume else {
        'last_completed_epoch': -1, 'best_loss': float('inf'), 'best_model_epoch': -1}
    start_epoch = status['last_completed_epoch'] + 1
    best_loss = status['best_loss']
    best_epoch = status['best_model_epoch']

    if args.resume and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")

    if start_epoch >= args.epochs:
        print(f"  Already trained for {args.epochs} epochs. Skipping.")
    else:
        # SMP training utilities
        train_epoch = smp.utils.train.TrainEpoch(
            model, loss=loss_fn, metrics=metrics,
            optimizer=optimizer, device=device, verbose=True)
        valid_epoch = smp.utils.train.ValidEpoch(
            model, loss=loss_fn, metrics=metrics,
            device=device, verbose=True)

        t0 = time.time()
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            # Extract validation loss (key depends on loss function name)
            val_loss = [v for k, v in valid_logs.items() if k != 'miou'][0]

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"  -> Best model saved (loss: {best_loss:.4f})")

            save_status(model_dir, epoch, best_loss, best_epoch)

        elapsed = time.time() - t0
        print(f"\nTraining done in {elapsed/60:.1f} min. Best epoch: {best_epoch+1}")

    # Evaluate on test set
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    results = evaluate_model(model, test_loader, device)

    print(f"\nTest results:")
    for i, iou in enumerate(results['ious']):
        print(f"  IoU class {i}: {iou*100:.2f}%")
    print(f"  mIoU:  {results['mIoU']*100:.2f}%")
    print(f"  fwIoU: {results['fwIoU']*100:.2f}%")

    # Save results
    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'dataset': dataset,
            'model': model_name,
            'encoder': encoder,
            'loss': loss_name,
            'epochs': args.epochs,
            'best_epoch': best_epoch + 1,
            'ious': [float(x) for x in results['ious']],
            'mIoU': float(results['mIoU']),
            'fwIoU': float(results['fwIoU']),
        }, f, indent=2)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # Determine combinations to train
    if args.best_only:
        combinations = BEST_MODELS
    elif args.all:
        combinations = ALL_COMBINATIONS
    elif args.all_models:
        if not args.dataset:
            print("Error: --all-models requires --dataset")
            sys.exit(1)
        combinations = [(args.dataset, m, e) for m in MODELS for e in ENCODERS]
    elif args.dataset and args.model:
        combinations = [(args.dataset, args.model, args.encoder)]
    else:
        print("Error: specify --dataset + --model, or use --all / --best-only / --all-models")
        sys.exit(1)

    print(f"Training {len(combinations)} model(s)\n")

    all_results = []
    for i, (ds, model, enc) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]")
        try:
            result = train_single(ds, model, enc, args.loss, args)
            all_results.append((ds, model, enc, result))
        except Exception as e:
            print(f"ERROR: {model}+{enc} on {ds}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Dataset':<12} {'Model':<18} {'Encoder':<22} {'mIoU':>8}")
        print('-' * 70)
        for ds, model, enc, result in all_results:
            print(f"{ds:<12} {model:<18} {enc:<22} {result['mIoU']*100:>7.2f}%")


if __name__ == '__main__':
    main()
