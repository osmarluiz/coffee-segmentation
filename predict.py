#!/usr/bin/env python
"""
Run inference on large geospatial images using sliding window approach.

Usage examples:
    # Predict on a single image:
    python predict.py --image path/to/image.tif --output predictions/result.tif

    # Specify model explicitly:
    python predict.py --image path/to/image.tif --output result.tif \
        --model Segformer --encoder efficientnet-b7 --channels 48 \
        --weights models/Planet_GT6/Segformer_efficientnet-b7/best_model.pth

    # Adjust sliding window parameters:
    python predict.py --image path/to/image.tif --output result.tif \
        --window 512 --stride 64 --batch-size 32
"""

import argparse
import os
import sys
import math

import torch
import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp


NUM_CLASSES = 3


def parse_args():
    p = argparse.ArgumentParser(description='Run segmentation inference on large images')

    # Input/Output
    p.add_argument('--image', type=str, required=True, help='Input geospatial image')
    p.add_argument('--output', type=str, required=True, help='Output prediction GeoTIFF')

    # Model
    p.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    p.add_argument('--model', type=str, default='Segformer', help='Model architecture')
    p.add_argument('--encoder', type=str, default='efficientnet-b7', help='Encoder backbone')
    p.add_argument('--channels', type=int, required=True,
                   help='Number of input channels (48=Planet, 30=VH/VV, 108=Combined)')

    # Sliding window
    p.add_argument('--window', type=int, default=512, help='Window size (default: 512)')
    p.add_argument('--stride', type=int, default=64, help='Stride (default: 64)')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    p.add_argument('--chunk-size', type=int, default=2560,
                   help='Chunk size for processing large images (default: 2560)')

    p.add_argument('--device', type=str, default=None)

    return p.parse_args()


def read_chunk(img_path, x_start, y_start, x_size, y_size):
    """Read a chunk from a geospatial raster."""
    with rasterio.open(img_path) as src:
        window = rasterio.windows.Window(x_start, y_start, x_size, y_size)
        chunk = src.read(window=window).astype('float32')
        transform = src.window_transform(window)
    return chunk, transform


def process_batch(batch, model, device):
    """Run inference on a batch of image windows."""
    with torch.no_grad():
        tensor = torch.from_numpy(np.stack(batch)).to(device).float()
        out = model(tensor)
        if isinstance(out, dict):
            out = out["semantic"]
        preds = out.cpu().numpy()
    return preds


def sliding_window(chunk, stride, window, model, device, batch_size=32):
    """Apply sliding window inference on a single chunk."""
    _, height, width = chunk.shape
    img_final = np.zeros((NUM_CLASSES, height, width), dtype='float32')

    model.eval()
    batch = []
    batch_coords = []

    for row in range(0, height - window + 1, stride):
        for col in range(0, width - window + 1, stride):
            tile = chunk[:, row:row+window, col:col+window]
            batch.append(tile)
            batch_coords.append((row, col))

            if len(batch) == batch_size:
                preds = process_batch(batch, model, device)
                for (r, c), pred in zip(batch_coords, preds):
                    img_final[:, r:r+window, c:c+window] += pred
                batch.clear()
                batch_coords.clear()

    if batch:
        preds = process_batch(batch, model, device)
        for (r, c), pred in zip(batch_coords, preds):
            img_final[:, r:r+window, c:c+window] += pred

    return np.argmax(img_final, axis=0).astype(np.uint8)


def predict(img_path, model, device, window, stride, batch_size, chunk_size):
    """Process a large image in chunks with sliding window inference."""
    with rasterio.open(img_path) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()
        crs = src.crs

    num_chunks_x = math.ceil(width / chunk_size)
    num_chunks_y = math.ceil(height / chunk_size)

    # Full output mosaic
    full_pred = np.zeros((height, width), dtype=np.uint8)

    total = num_chunks_x * num_chunks_y
    print(f"Image: {width}x{height}, processing in {total} chunks ({chunk_size}x{chunk_size})")

    for j in tqdm(range(num_chunks_y), desc='Rows'):
        for i in range(num_chunks_x):
            x_start = i * chunk_size
            y_start = j * chunk_size
            x_size = min(chunk_size, width - x_start)
            y_size = min(chunk_size, height - y_start)

            chunk, _ = read_chunk(img_path, x_start, y_start, x_size, y_size)
            pred = sliding_window(chunk, stride, window, model, device, batch_size)
            full_pred[y_start:y_start+y_size, x_start:x_start+x_size] = pred

    return full_pred, meta, crs


def save_prediction(prediction, meta, crs, output_path):
    """Save prediction as GeoTIFF with geospatial metadata."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'LZW',
        'height': prediction.shape[0],
        'width': prediction.shape[1],
        'crs': crs,
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction, 1)


def main():
    args = parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # Load model
    print(f"Loading {args.model}+{args.encoder} ({args.channels} channels)...")
    model = getattr(smp, args.model)(
        encoder_name=args.encoder,
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation='softmax',
        in_channels=args.channels,
    )
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Run inference
    prediction, meta, crs = predict(
        args.image, model, args.device,
        args.window, args.stride, args.batch_size, args.chunk_size)

    # Save
    save_prediction(prediction, meta, crs, args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == '__main__':
    main()
