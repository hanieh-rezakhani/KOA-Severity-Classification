"""Model evaluation helpers."""
from __future__ import annotations
import torch
from .metrics import evaluate_predictions

def predict(model, loader, device='cpu'):
    device = torch.device(device); model.to(device).eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            p = torch.softmax(logits, dim=1).cpu()
            probs.append(p); ys.extend(labels.numpy()); ps.extend(p.argmax(1).numpy())
    probabilities = torch.cat(probs).numpy()
    return {'y_true': ys, 'y_pred': ps, 'probabilities': probabilities,
            'metrics': evaluate_predictions(ys, ps, probabilities, probabilities.shape[1])}
