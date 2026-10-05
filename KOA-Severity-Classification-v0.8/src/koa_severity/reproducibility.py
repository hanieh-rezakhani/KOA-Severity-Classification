"""Reproducibility helpers shared by experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 2026, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    Deterministic execution can reduce performance and does not guarantee
    bit-for-bit reproducibility across different hardware/software stacks.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
