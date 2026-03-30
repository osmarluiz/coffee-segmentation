"""
Data utilities for coffee segmentation.
Handles dataset loading and preprocessing with geospatial support.
"""

import os
import numpy as np
import torch
import imageio.v2 as imageio
import rasterio
import random
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple


class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading multi-channel TIFF images and PNG masks.

    Handles:
    - Multi-channel TIFF loading via rasterio (supports up to 108 channels)
    - Label remapping: original labels 2->1, 3->2
    - Synchronized augmentation for image-mask pairs
    """

    def __init__(self, image_paths, target_paths, transform=None, transform_label=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.transform_label = transform_label

        if len(image_paths) != len(target_paths):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(target_paths)} masks")

    def __getitem__(self, index):
        # Load multi-channel image with rasterio
        with rasterio.open(self.image_paths[index]) as src:
            image = src.read().astype(np.float32)
            image = np.moveaxis(image, 0, -1)  # (C,H,W) -> (H,W,C)

        # Load mask
        mask = imageio.imread(self.target_paths[index]).astype(np.int64)

        # Remap labels: 0->0 (background), 2->1 (coffee), 3->2 (eucalyptus)
        mask = np.where(mask == 2, 1, mask)
        mask = np.where(mask == 3, 2, mask)

        # Synchronized random seed for consistent augmentation
        seed = np.random.randint(2147483647)

        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)

        if self.transform_label:
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform_label(mask)
            mask = mask.squeeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


def get_transforms(augment=True):
    """Get image and mask transforms. Returns (image_tf, mask_tf)."""
    if augment:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        return tf, tf
    else:
        return transforms.ToTensor(), None
