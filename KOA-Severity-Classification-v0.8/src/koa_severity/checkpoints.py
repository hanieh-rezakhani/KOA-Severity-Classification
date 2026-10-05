"""Checkpoint persistence and validation."""
from __future__ import annotations
from pathlib import Path
import torch

def load_checkpoint(path, model, device='cpu', strict=True):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict): raise TypeError('Checkpoint does not contain a state dict.')
    missing, unexpected = model.load_state_dict(state, strict=strict)
    model.to(device)
    return payload if isinstance(payload, dict) else {}, missing, unexpected
