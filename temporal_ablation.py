#!/usr/bin/env python
"""
Temporal Ablation Study for Coffee Plantation Segmentation

Addresses reviewer R2.7 (revision 2): "compare 3-month, 6-month, 9-month, and
12-month temporal inputs, and report the resulting mIoU and class-specific IoU
metrics", with attention to "the most informative phenological windows for
coffee mapping".

Trains SegFormer + EfficientNet-B7 (the best paper configuration) on Planet
NICFI subsets, selecting different consecutive month windows. Each Planet tile
has 48 bands = 4 spectral bands (B,G,R,NIR) x 12 monthly composites. Bands are
selected at runtime via rasterio's band indexing (no disk duplication).

Each window is trained for 100 epochs with the same hyperparameters as the
paper baseline (Adam lr=1e-3, weighted CE [1, 15, 35], batch 8, 100 epochs).
The model checkpoint is saved by val_loss (paper criterion) AND by val_mIoU
(auxiliary criterion). Both are evaluated on the test set.

Quick start:
    python temporal_ablation.py --window 12-full --amp bfloat16
    python temporal_ablation.py --all --amp bfloat16

Resume from crash (watchdog-friendly): rerun the same command. The script
skips windows whose results are already in temporal_ablation_results.csv and
resumes mid-window from training_status.json.
"""

import argparse
import os
import sys
import json
import time
import datetime

import numpy as np
import pandas as pd
import torch
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp
from src.data_utils import CustomDataset


# ─── Constants ───────────────────────────────────────────────────────────────

PLANET_DIR = 'DATASET/Planet_GT6'
OUTPUT_DIR = 'models/temporal_ablation'
NUM_CLASSES = 3
CLASS_WEIGHTS = [1.0, 15.0, 35.0]
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_TRAIN = 8
BATCH_EVAL = 32

# Months are 1-indexed (Jan=1, Dec=12). Each month has 4 bands [B, G, R, NIR]
# at indices [(m-1)*4 : m*4].
WINDOWS = [
    # 3-month consecutive windows (quarterly):
    {'name': '3-cons-Jan',  'months': [1, 2, 3],          'desc': 'Q1 Jan-Mar'},
    {'name': '3-cons-Apr',  'months': [4, 5, 6],          'desc': 'Q2 Apr-Jun'},
    {'name': '3-cons-Jul',  'months': [7, 8, 9],          'desc': 'Q3 Jul-Sep (harvest)'},
    {'name': '3-cons-Oct',  'months': [10, 11, 12],       'desc': 'Q4 Oct-Dec'},
    # 6-month consecutive windows:
    {'name': '6-cons-1H',   'months': [1, 2, 3, 4, 5, 6], 'desc': 'First semester Jan-Jun'},
    {'name': '6-cons-Apr',  'months': [4, 5, 6, 7, 8, 9], 'desc': 'Apr-Sep (maturation+harvest)'},
    {'name': '6-cons-2H',   'months': [7, 8, 9, 10, 11, 12], 'desc': 'Second semester Jul-Dec'},
    # 9-month consecutive windows:
    {'name': '9-cons',      'months': [1, 2, 3, 4, 5, 6, 7, 8, 9],     'desc': 'Jan-Sep'},
    {'name': '9-cons-Dec',  'months': [4, 5, 6, 7, 8, 9, 10, 11, 12],  'desc': 'Apr-Dec'},
    # 12-month full baseline (paper config + AMP for internal consistency):
    {'name': '12-full',     'months': list(range(1, 13)), 'desc': 'Full year baseline'},
]

WINDOWS_BY_NAME = {w['name']: w for w in WINDOWS}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def months_to_band_indices(months):
    """Map a list of months (1-12) to Planet band indices (0-47)."""
    indices = []
    for m in months:
        start = (m - 1) * 4
        indices.extend([start, start + 1, start + 2, start + 3])
    return indices


def log(msg, log_file):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_status(model_dir):
    f = os.path.join(model_dir, 'training_status.json')
    if os.path.exists(f):
        s = json.load(open(f))
        s.setdefault('best_miou', -float('inf'))
        s.setdefault('best_miou_epoch', -1)
        return s
    return {
        'last_completed_epoch': -1,
        'best_loss': float('inf'),
        'best_model_epoch': -1,
        'best_miou': -float('inf'),
        'best_miou_epoch': -1,
    }


def save_status(model_dir, epoch, best_loss, best_epoch, best_miou, best_miou_epoch):
    f = os.path.join(model_dir, 'training_status.json')
    json.dump({
        'last_completed_epoch': epoch,
        'best_loss': best_loss,
        'best_model_epoch': best_epoch,
        'best_miou': best_miou,
        'best_miou_epoch': best_miou_epoch,
    }, open(f, 'w'))


def resolve_amp_dtype(amp_flag, device):
    """Resolve --amp flag to a torch dtype or None."""
    if device != 'cuda':
        return None
    if amp_flag == 'none':
        return None
    if amp_flag == 'bfloat16':
        return torch.bfloat16
    if amp_flag == 'float16':
        return torch.float16
    return None


# ─── Evaluation (confusion-matrix based) ─────────────────────────────────────

def evaluate_model(model, data_loader, device, num_classes=3, amp_dtype=None):
    """Test evaluation using confusion matrix for accurate per-class IoU/mIoU/fwIoU."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for inp, lab in data_loader:
            inp = inp.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(inp)
                outputs = outputs.float()
            else:
                outputs = model(inp)
            _, preds = torch.max(outputs, dim=1)
            cm += confusion_matrix(
                lab.view(-1).cpu().numpy(),
                preds.view(-1).cpu().numpy(),
                labels=range(num_classes),
            )
    ious = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        ious.append(TP / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0)
    mIoU = float(np.mean(ious))
    fwIoU = float(np.sum(cm.sum(axis=1) / cm.sum() * np.array(ious)))
    return {'ious': [float(x) for x in ious], 'mIoU': mIoU, 'fwIoU': fwIoU}


# ─── Single-window training ──────────────────────────────────────────────────

def train_one_window(window, args, device, log_file, results_csv):
    name = window['name']
    months = window['months']
    n_months = len(months)
    in_channels = n_months * 4
    band_indices = months_to_band_indices(months)

    log('=' * 70, log_file)
    log(f'WINDOW: {name} | Months: {months} | Channels: {in_channels}', log_file)
    log(f'Description: {window["desc"]}', log_file)
    log('=' * 70, log_file)

    model_dir = os.path.join(args.output_dir,
                             f'Planet_Segformer_efficientnet-b7_{name}')
    os.makedirs(model_dir, exist_ok=True)

    # ── Data loaders ─────────────────────────────────────────────────────────
    planet_dir = args.planet_dir
    train_imgs = sorted(glob(os.path.join(planet_dir, 'image_train/*.tiff')))
    val_imgs   = sorted(glob(os.path.join(planet_dir, 'image_val/*.tiff')))
    test_imgs  = sorted(glob(os.path.join(planet_dir, 'image_test/*.tiff')))
    train_masks = sorted(glob(os.path.join(planet_dir, 'class_train/*.png')))
    val_masks   = sorted(glob(os.path.join(planet_dir, 'class_val/*.png')))
    test_masks  = sorted(glob(os.path.join(planet_dir, 'class_test/*.png')))

    log(f'Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}',
        log_file)

    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    no_aug = transforms.ToTensor()

    train_ds = CustomDataset(train_imgs, train_masks,
                             transform=aug, transform_label=aug,
                             band_indices=band_indices)
    val_ds   = CustomDataset(val_imgs, val_masks,
                             transform=no_aug, transform_label=None,
                             band_indices=band_indices)
    test_ds  = CustomDataset(test_imgs, test_masks,
                             transform=no_aug, transform_label=None,
                             band_indices=band_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_EVAL, shuffle=False,
                              num_workers=min(args.num_workers, 2), pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    test_loader  = DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False,
                              num_workers=min(args.num_workers, 2), pin_memory=True,
                              persistent_workers=(args.num_workers > 0))

    # ── Model, loss, optimizer ───────────────────────────────────────────────
    model = smp.Segformer(
        encoder_name='efficientnet-b7',
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation='softmax',
        in_channels=in_channels,
    ).to(device)

    weights = torch.tensor(CLASS_WEIGHTS, device=device)
    loss_fn = smp.utils.losses.CrossEntropyLoss(weight=weights).to(device)
    metrics = [smp.utils.metrics.mIoU()]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Resume ───────────────────────────────────────────────────────────────
    status = load_status(model_dir)
    start_epoch = status['last_completed_epoch'] + 1
    best_loss = status['best_loss']
    best_loss_epoch = status['best_model_epoch']
    best_miou = status['best_miou']
    best_miou_epoch = status['best_miou_epoch']

    cp_loss = os.path.join(model_dir, 'best_model_loss.pth')
    cp_miou = os.path.join(model_dir, 'best_model_miou.pth')
    cp_last = os.path.join(model_dir, 'last_model.pth')
    cp_legacy = os.path.join(model_dir, 'best_model.pth')

    if start_epoch > 0:
        for cp in [cp_last, cp_loss, cp_legacy]:
            if os.path.exists(cp):
                model.load_state_dict(torch.load(cp, weights_only=True, map_location=device))
                log(f'Resumed from epoch {start_epoch} via {os.path.basename(cp)}',
                    log_file)
                break

    # ── AMP-enabled SMP TrainEpoch/ValidEpoch ────────────────────────────────
    amp_dtype = resolve_amp_dtype(args.amp, device)
    if amp_dtype is not None:
        log(f'AMP enabled: autocast(dtype={amp_dtype}) — no GradScaler', log_file)

    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss_fn, metrics=metrics,
        optimizer=optimizer, device=device, verbose=True,
        amp_dtype=amp_dtype,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss_fn, metrics=metrics,
        device=device, verbose=True,
        amp_dtype=amp_dtype,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    if start_epoch >= NUM_EPOCHS:
        log(f'[{name}] Already trained for {NUM_EPOCHS} epochs. Skipping training.',
            log_file)
    else:
        for epoch in range(start_epoch, NUM_EPOCHS):
            log(f'[{name}] Epoch {epoch + 1}/{NUM_EPOCHS}', log_file)
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            # Loss key depends on loss name; mIoU key is 'miou' (from SMP).
            v_loss = [v for k, v in valid_logs.items() if k != 'miou'][0]
            v_miou = valid_logs.get('miou', 0.0)
            log(f'[{name}] Epoch {epoch+1}: '
                f'train_loss={train_logs[next(iter(train_logs))]:.4f}, '
                f'val_loss={v_loss:.4f}, val_mIoU={v_miou:.4f}', log_file)

            if v_loss < best_loss:
                best_loss = v_loss
                best_loss_epoch = epoch
                torch.save(model.state_dict(), cp_loss)
                log(f'[{name}] New best LOSS at epoch {epoch+1}: {v_loss:.4f}',
                    log_file)

            if v_miou > best_miou:
                best_miou = v_miou
                best_miou_epoch = epoch
                torch.save(model.state_dict(), cp_miou)
                log(f'[{name}] New best mIoU at epoch {epoch+1}: {v_miou:.4f}',
                    log_file)

            # Always save last weights to allow crash-safe resume even when
            # best loss/mIoU lag the actual training state.
            torch.save(model.state_dict(), cp_last)
            save_status(model_dir, epoch, best_loss, best_loss_epoch,
                        best_miou, best_miou_epoch)

    # ── Final test evaluation on both checkpoints ────────────────────────────
    cp_loss_resolved = cp_loss if os.path.exists(cp_loss) else cp_legacy
    log(f'[{name}] Loading best-by-LOSS (epoch {best_loss_epoch+1}) for test eval',
        log_file)
    model.load_state_dict(torch.load(cp_loss_resolved, weights_only=True,
                                      map_location=device))
    results_loss = evaluate_model(model, test_loader, device,
                                  num_classes=NUM_CLASSES, amp_dtype=amp_dtype)
    log(f'[{name}] TEST (best-by-LOSS): '
        f'mIoU={results_loss["mIoU"]*100:.2f}% '
        f'IoU0={results_loss["ious"][0]*100:.2f}% '
        f'IoU1={results_loss["ious"][1]*100:.2f}% '
        f'IoU2={results_loss["ious"][2]*100:.2f}%', log_file)

    results_miou = None
    if best_miou_epoch >= 0 and os.path.exists(cp_miou):
        log(f'[{name}] Loading best-by-mIoU (epoch {best_miou_epoch+1}) for test eval',
            log_file)
        model.load_state_dict(torch.load(cp_miou, weights_only=True,
                                          map_location=device))
        results_miou = evaluate_model(model, test_loader, device,
                                      num_classes=NUM_CLASSES, amp_dtype=amp_dtype)
        log(f'[{name}] TEST (best-by-mIoU): '
            f'mIoU={results_miou["mIoU"]*100:.2f}% '
            f'IoU0={results_miou["ious"][0]*100:.2f}% '
            f'IoU1={results_miou["ious"][1]*100:.2f}% '
            f'IoU2={results_miou["ious"][2]*100:.2f}%', log_file)

    # ── Persist results ──────────────────────────────────────────────────────
    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'window': name,
            'months': months,
            'in_channels': in_channels,
            'best_loss_epoch': best_loss_epoch + 1,
            'best_val_loss': best_loss,
            'test_by_loss': results_loss,
            'best_miou_epoch': best_miou_epoch + 1 if best_miou_epoch >= 0 else None,
            'best_val_miou': best_miou if best_miou > -float('inf') else None,
            'test_by_miou': results_miou,
        }, f, indent=2)

    rows = [{
        'Window': name, 'Months': str(months), 'N_Months': n_months,
        'In_Channels': in_channels, 'Selection': 'val_loss',
        'Best_Epoch': best_loss_epoch + 1, 'Best_Val_Loss': best_loss,
        'Best_Val_mIoU': None,
        'IoU_0': results_loss['ious'][0] * 100,
        'IoU_1': results_loss['ious'][1] * 100,
        'IoU_2': results_loss['ious'][2] * 100,
        'mIoU': results_loss['mIoU'] * 100,
        'fwIoU': results_loss['fwIoU'] * 100,
    }]
    if results_miou is not None:
        rows.append({
            'Window': name, 'Months': str(months), 'N_Months': n_months,
            'In_Channels': in_channels, 'Selection': 'val_miou',
            'Best_Epoch': best_miou_epoch + 1, 'Best_Val_Loss': None,
            'Best_Val_mIoU': best_miou,
            'IoU_0': results_miou['ious'][0] * 100,
            'IoU_1': results_miou['ious'][1] * 100,
            'IoU_2': results_miou['ious'][2] * 100,
            'mIoU': results_miou['mIoU'] * 100,
            'fwIoU': results_miou['fwIoU'] * 100,
        })
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Temporal ablation for coffee segmentation (Planet only).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--window', type=str, default=None,
                   choices=list(WINDOWS_BY_NAME.keys()),
                   help='Train a single named window (e.g. 12-full).')
    p.add_argument('--all', action='store_true',
                   help='Train every window in the predefined list.')
    p.add_argument('--planet-dir', type=str, default=PLANET_DIR,
                   help='Path to Planet_GT6 directory (default: DATASET/Planet_GT6).')
    p.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                   help='Where to save models and results (default: models/temporal_ablation).')
    p.add_argument('--amp', type=str, default='bfloat16',
                   choices=['bfloat16', 'float16', 'none'],
                   help='Autocast dtype (default: bfloat16; matches main paper run).')
    p.add_argument('--num-workers', type=int, default=4,
                   help='DataLoader workers (default: 4).')
    p.add_argument('--device', type=str, default=None,
                   help='Device override (default: cuda if available).')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.all and not args.window:
        sys.exit('Please provide --window <name> or --all.')

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'ablation.log')
    results_csv = os.path.join(args.output_dir, 'temporal_ablation_results.csv')

    log('=' * 70, log_file)
    log('TEMPORAL ABLATION STUDY - Planet only, SegFormer + EfficientNet-B7',
        log_file)
    log('=' * 70, log_file)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}', log_file)
    if device == 'cuda':
        log(f'GPU: {torch.cuda.get_device_name(0)}', log_file)

    # Load existing results for skip-on-restart behaviour.
    all_results = []
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        all_results = df.to_dict('records')
        log(f'Loaded {len(all_results)} existing result rows', log_file)
    completed = {r['Window'] for r in all_results}

    windows_to_train = (
        WINDOWS if args.all else [WINDOWS_BY_NAME[args.window]]
    )

    for w in windows_to_train:
        if w['name'] in completed:
            log(f'SKIP {w["name"]} (already in results CSV)', log_file)
            continue
        try:
            rows = train_one_window(w, args, device, log_file, results_csv)
            all_results.extend(rows)
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
            log(f'Saved results to {results_csv}', log_file)
        except Exception as e:
            import traceback
            log(f'ERROR on window {w["name"]}: {e}', log_file)
            log(traceback.format_exc(), log_file)
            # Move on so other windows can still run; watchdog can relaunch
            # later to retry from the last successful epoch.
            continue

    log('=' * 70, log_file)
    log('TEMPORAL ABLATION COMPLETE', log_file)
    log('=' * 70, log_file)


if __name__ == '__main__':
    main()
