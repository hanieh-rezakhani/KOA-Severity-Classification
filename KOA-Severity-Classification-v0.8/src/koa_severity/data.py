"""Dataset/path helpers for the five-class KOA dataset."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

CLASS_NAMES = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


def get_dataset_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the dataset root without downloading or embedding credentials."""
    raw = path or os.getenv("KOA_DATASET_PATH")
    if not raw:
        raise ValueError(
            "Dataset path is not configured. Set KOA_DATASET_PATH or pass path=..."
        )
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    return root


def split_dir(dataset_root: str | os.PathLike[str], split: str) -> Path:
    """Return a standard split directory such as train/val/test."""
    root = Path(dataset_root).expanduser().resolve()
    path = root / split
    if not path.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {path}")
    return path


def class_directories(
    dataset_root: str | os.PathLike[str], split: str
) -> dict[int, Path]:
    """Map numeric labels to existing ImageFolder-style class directories."""
    root = split_dir(dataset_root, split)
    return {label: root / str(label) for label in CLASS_NAMES}
