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
from segmentation_models_pytorch.decoders.mask2former.matcher import HungarianMatcher
from segmentation_models_pytorch.decoders.mask2former.criterion import SetCriterion
from segmentation_models_pytorch.decoders.mask2former.training import (
    Mask2FormerTrainEpoch,
    Mask2FormerValidEpoch,
)
from src.data_utils import CustomDataset
from src.loss_functions import MulticlassDiceLoss, MulticlassFocalLoss


# ─── Constants ───────────────────────────────────────────────────────────────

DATASETS = {
    'Combined': {'channels': 108, 'dir': 'Combined_GT6'},
    'Planet':   {'channels': 48,  'dir': 'Planet_GT6'},
    'VH':       {'channels': 30,  'dir': 'VH_GT6'},
    'VV':       {'channels': 30,  'dir': 'VV_GT6'},
}

MODELS = ['Unet', 'Linknet', 'UnetPlusPlus', 'PSPNet', 'DeepLabV3Plus', 'Segformer', 'Mask2Former']

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
    p.add_argument('--amp', type=str, default='auto',
                   choices=['auto', 'bfloat16', 'float16', 'none'],
                   help='Autocast dtype for forward pass. "auto" enables '
                        'bfloat16 for Mask2Former and disables it for the '
                        'paper-baseline models. bfloat16 dispenses GradScaler.')

    return p.parse_args()


# ─── AMP helper ──────────────────────────────────────────────────────────────

def _resolve_amp_dtype(amp_flag, model_name, device):
    """Resolve the --amp flag into a concrete torch dtype or None."""
    if device != 'cuda':
        return None
    if amp_flag == 'none':
        return None
    if amp_flag == 'bfloat16':
        return torch.bfloat16
    if amp_flag == 'float16':
        return torch.float16
    # 'auto': enable bf16 only for Mask2Former
    if model_name == 'Mask2Former':
        return torch.bfloat16
    return None


# ─── Model creation ──────────────────────────────────────────────────────────

def create_model(model_name, encoder_name, in_channels):
    """Create a segmentation model using SMP.

    Activation note: the other six architectures (the paper baseline) keep
    `activation='softmax'` in the segmentation head. This means the model
    output is softmax-normalised before being fed to `nn.CrossEntropyLoss`,
    which itself applies `log_softmax` internally — i.e. a double softmax.
    For those models it is a mild inefficiency that the paper tolerates.

    For Mask2Former it is fatal: the decoder's semantic output is already
    `einsum(softmax(class), sigmoid(mask))` summed over 100 queries, which at
    initialization yields very similar magnitudes across the 3 classes per
    pixel. A second softmax then collapses the distribution to ~uniform and
    the CE gradient effectively vanishes (loss stays pinned at log(3) and the
    model just predicts background everywhere). We therefore pass logits
    directly for Mask2Former, while preserving the paper's setup for the
    baseline models.
    """
    activation = None if model_name == 'Mask2Former' else 'softmax'
    extra_kwargs = {}
    if model_name == 'Mask2Former':
        # With Hungarian-matched training, unmatched queries learn the
        # no-object class and do not destabilise optimisation, so we can
        # keep the paper default (100). For a 3-class problem with 2-3 GT
        # instances per image, ~97 queries end up specialising on the
        # no-object class, which is fine.
        extra_kwargs['num_queries'] = 100
    return getattr(smp, model_name)(
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation=activation,
        in_channels=in_channels,
        **extra_kwargs,
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
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = model(images)
            if isinstance(out, dict):
                out = out["semantic"]
            preds = out.argmax(dim=1)

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
    # Cast to native Python types — numpy.float32 (from AverageValueMeter.mean)
    # is not JSON-serializable.
    with open(os.path.join(model_dir, 'training_status.json'), 'w') as f:
        json.dump({
            'last_completed_epoch': int(epoch),
            'best_loss': float(best_loss),
            'best_model_epoch': int(best_epoch),
        }, f)


def load_status(model_dir):
    path = os.path.join(model_dir, 'training_status.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            # Truncated / corrupted status (e.g. crash mid-write). Restart clean.
            print(f"  WARNING: training_status.json is corrupted ({e}); ignoring it.")
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
    last_path = os.path.join(model_dir, 'last_model.pth')

    # Data
    train_loader, val_loader, test_loader = load_data(
        args.data_dir, dataset, args.batch_size, args.num_workers)

    # Model
    model = create_model(model_name, encoder, in_channels)
    model.to(device)

    # Loss and metrics
    if model_name == 'Mask2Former':
        # Mask2Former uses Hungarian-matched set prediction, not a dense
        # per-pixel CE/Dice/Focal loss. Without it, queries have no
        # specialisation signal and the model collapses to predicting the
        # uniform class distribution (see project history). The CE weights
        # CLASS_WEIGHTS [1, 15, 35] from the paper are carried over into the
        # per-query classification head; the no-object class is down-weighted
        # via eos_coef.
        matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
        loss_fn = SetCriterion(
            num_classes=NUM_CLASSES,
            matcher=matcher,
            weight_dict={'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0},
            eos_coef=0.1,
            aux_loss=True,
            class_weight=CLASS_WEIGHTS,
        )
        loss_fn.to(device)
        print(f"  Loss: SetCriterion (Hungarian) with weight_dict "
              f"loss_ce=2.0, loss_mask=5.0, loss_dice=5.0; "
              f"class_weight={CLASS_WEIGHTS}, eos_coef=0.1; aux at 9 layers")
    else:
        loss_fn = get_loss_fn(loss_name, device)
    metrics = [smp.utils.metrics.mIoU()]

    # Optimizer
    # Mask2Former is far more sensitive to LR than the CNN-based baselines.
    # With Adam(lr=1e-3) and random init, the first optimizer step overshoots
    # into the uniform-prediction attractor (val_loss == log(3)) and never
    # escapes. The original Mask2Former paper trains with AdamW, lr=1e-4 and
    # weight_decay=0.05; we adopt that recipe here. The other six baseline
    # architectures keep the paper's Adam(lr=1e-3) for direct comparison with
    # the published results.
    if model_name == 'Mask2Former':
        m2f_lr = args.lr if args.lr != 0.001 else 1e-4  # 1e-4 unless user overrode
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=m2f_lr, weight_decay=0.05)
        print(f"  Optimizer: AdamW(lr={m2f_lr}, weight_decay=0.05) "
              f"[Mask2Former-specific recipe]")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    status = load_status(model_dir) if args.resume else {
        'last_completed_epoch': -1, 'best_loss': float('inf'), 'best_model_epoch': -1}
    start_epoch = status['last_completed_epoch'] + 1
    best_loss = status['best_loss']
    best_epoch = status['best_model_epoch']

    if args.resume:
        # Prefer last_model.pth (latest checkpointed weights) over
        # best_model.pth: best_model only updates when val_loss improves, so
        # after a crash it can be many epochs behind the actual training
        # state. last_model is always the most recent epoch's weights.
        if os.path.exists(last_path):
            model.load_state_dict(torch.load(last_path, map_location=device))
            print(f"  Resumed from epoch {start_epoch} (last_model.pth), "
                  f"best loss: {best_loss:.4f}")
        elif os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  Resumed from epoch {start_epoch} (best_model.pth — "
                  f"last_model not found), best loss: {best_loss:.4f}")

    if start_epoch >= args.epochs:
        print(f"  Already trained for {args.epochs} epochs. Skipping.")
    else:
        # SMP training utilities with optional AMP autocast (bfloat16 by default
        # for Mask2Former, disabled for the paper-baseline models).
        amp_dtype = _resolve_amp_dtype(args.amp, model_name, device)
        if amp_dtype is not None:
            print(f"  AMP enabled: autocast(dtype={amp_dtype}) — no GradScaler.")

        if model_name == 'Mask2Former':
            # Mask2Former wraps a richer interface: the model returns a dict
            # ({pred_logits, pred_masks, aux_outputs, semantic}), and the loss
            # is a Hungarian-matched SetCriterion that runs over all decoder
            # layers (deep supervision).
            train_epoch = Mask2FormerTrainEpoch(
                model, criterion=loss_fn, metrics=metrics,
                optimizer=optimizer, device=device, verbose=True,
                amp_dtype=amp_dtype)
            valid_epoch = Mask2FormerValidEpoch(
                model, criterion=loss_fn, metrics=metrics,
                device=device, verbose=True,
                amp_dtype=amp_dtype)
        else:
            train_epoch = smp.utils.train.TrainEpoch(
                model, loss=loss_fn, metrics=metrics,
                optimizer=optimizer, device=device, verbose=True,
                amp_dtype=amp_dtype)
            valid_epoch = smp.utils.train.ValidEpoch(
                model, loss=loss_fn, metrics=metrics,
                device=device, verbose=True,
                amp_dtype=amp_dtype)

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

            # Always save the latest weights too so an OS reboot or crash
            # mid-training can be resumed without losing the more-trained
            # state (best_model only updates on val_loss improvement, which
            # can lag many epochs behind the actual learning).
            torch.save(model.state_dict(), last_path)

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
