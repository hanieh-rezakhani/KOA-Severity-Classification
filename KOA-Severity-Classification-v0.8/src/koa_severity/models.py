"""Model/checkpoint helpers shared by evaluation scripts."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import torch
import torch.nn as nn
from torchvision import models

_COMMON_PREFIXES = ("backbone.", "model.", "encoder.", "_orig_mod.", "module.")

def clean_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove common wrapper prefixes from checkpoint keys."""
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in _COMMON_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
                    break
        cleaned[name] = value
    return cleaned

def load_resnet18_checkpoint(weights_path: str | Path, num_classes: int = 5,
                             device: str | torch.device = "cpu", strict: bool = True):
    """Load a ResNet18 classifier and optionally fail on incomplete checkpoints."""
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state-dict mapping.")
    state_dict = clean_state_dict_keys(state_dict)
    out_features = state_dict.get("fc.weight", torch.empty(num_classes, 512)).shape[0]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(out_features))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.to(device).eval()
    return model, missing, unexpected
