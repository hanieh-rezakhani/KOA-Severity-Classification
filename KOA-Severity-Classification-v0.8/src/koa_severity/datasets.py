"""Reusable ImageFolder dataset and DataLoader construction."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .data import split_dir

@dataclass(frozen=True)
class LoaderConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    weighted_sampling: bool = False

class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int,int] = (8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def __call__(self, img):
        import cv2
        from PIL import Image
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return Image.fromarray(clahe.apply(arr))

def image_transforms(image_size: int = 224, train: bool = False, clahe: bool = True):
    ops = [transforms.Resize((image_size, image_size))]
    if clahe: ops.append(CLAHETransform())
    ops.append(transforms.Grayscale(num_output_channels=3))
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.03,0.03), scale=(0.95,1.05)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    return transforms.Compose(ops)

def make_imagefolder_loader(dataset_root, split: str, config: LoaderConfig = LoaderConfig(), train: bool = False):
    directory = split_dir(dataset_root, split)
    dataset = datasets.ImageFolder(directory, transform=image_transforms(config.image_size, train=train))
    sampler = None
    shuffle = train
    if train and config.weighted_sampling:
        targets = np.asarray(dataset.targets)
        counts = np.bincount(targets, minlength=len(dataset.classes))
        if np.any(counts == 0):
            raise ValueError(f"Cannot build weighted sampler: empty class in {dataset.classes}")
        weights = torch.as_tensor(1.0 / counts[targets], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
                        num_workers=config.num_workers, pin_memory=config.pin_memory)
    return dataset, loader
