"""Small, dependency-free experiment configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    num_classes: int = 5
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-4
    weighted_sampling: bool = True
    seed: int = 2026
