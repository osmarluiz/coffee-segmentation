#!/usr/bin/env python
"""
Prepare the Combined dataset by concatenating VH, VV, and optionally Planet bands.

The Combined dataset stacks all available modalities into a single multi-channel
GeoTIFF per tile:
    - Planet only (if available): 48 bands (4 bands x 12 months)
    - VH: 30 bands (Sentinel-1 VH polarization, monthly composites)
    - VV: 30 bands (Sentinel-1 VV polarization, monthly composites)
    - Combined = Planet + VV + VH = 108 bands (or VV + VH = 60 bands without Planet)

Usage examples:
    # Combine VH + VV only (no Planet):
    python prepare_combined.py \
        --vh DATASET/VH_GT6 \
        --vv DATASET/VV_GT6 \
        --output DATASET/Combined_GT6

    # Combine Planet + VV + VH:
    python prepare_combined.py \
        --planet DATASET/Planet_GT6 \
        --vh DATASET/VH_GT6 \
        --vv DATASET/VV_GT6 \
        --output DATASET/Combined_GT6
"""

import argparse
import os
import numpy as np
import rasterio
from glob import glob
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description='Combine multi-modal datasets')
    p.add_argument('--planet', type=str, default=None,
                   help='Planet_GT6 directory (optional)')
    p.add_argument('--vh', type=str, required=True,
                   help='VH_GT6 directory')
    p.add_argument('--vv', type=str, required=True,
                   help='VV_GT6 directory')
    p.add_argument('--output', type=str, required=True,
                   help='Output Combined_GT6 directory')
    return p.parse_args()


def combine_and_save(image_paths, output_path, reference_path):
    """Combine multiple images by stacking bands and save as GeoTIFF."""
    arrays = []
    for path in image_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read().astype(np.float32))

    combined = np.concatenate(arrays, axis=0)  # (C, H, W)

    with rasterio.open(reference_path) as ref:
        transform = ref.transform
        crs = ref.crs

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=combined.shape[1], width=combined.shape[2],
        count=combined.shape[0], dtype=combined.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(combined)


def process_split(split_name, sources, output_dir):
    """Process one split (train/val/test)."""
    img_subdir = f"image_{split_name}"
    out_dir = os.path.join(output_dir, img_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Get file lists from each source
    file_lists = []
    for src_dir in sources:
        files = sorted(glob(os.path.join(src_dir, img_subdir, '*.tiff')))
        if not files:
            print(f"  Warning: no .tiff files in {src_dir}/{img_subdir}/")
            return 0
        file_lists.append(files)

    # Check all sources have same number of files
    counts = [len(fl) for fl in file_lists]
    if len(set(counts)) > 1:
        print(f"  Warning: mismatched file counts in {split_name}: {counts}")
        return 0

    n_files = counts[0]
    for idx in tqdm(range(n_files), desc=f"  {split_name}"):
        paths = [fl[idx] for fl in file_lists]
        basename = os.path.basename(paths[0])
        output_path = os.path.join(out_dir, basename)
        combine_and_save(paths, output_path, reference_path=paths[0])

    return n_files


def main():
    args = parse_args()

    # Build source list: Planet (optional) + VV + VH
    sources = []
    source_names = []
    if args.planet:
        sources.append(args.planet)
        source_names.append('Planet')
    sources.extend([args.vv, args.vh])
    source_names.extend(['VV', 'VH'])

    # Count total channels
    total_channels = 0
    for src_dir in sources:
        sample = sorted(glob(os.path.join(src_dir, 'image_train/*.tiff')))
        if sample:
            with rasterio.open(sample[0]) as src:
                total_channels += src.count

    print(f"Combining: {' + '.join(source_names)} = {total_channels} channels")
    print(f"Output: {args.output}")

    os.makedirs(args.output, exist_ok=True)

    for split in ['train', 'val', 'test']:
        n = process_split(split, sources, args.output)
        print(f"  {split}: {n} images")

    print("\nDone!")


if __name__ == '__main__':
    main()
