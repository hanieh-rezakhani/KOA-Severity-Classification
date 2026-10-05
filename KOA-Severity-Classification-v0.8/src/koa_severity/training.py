"""Generic supervised PyTorch training loop for image classifiers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from .metrics import evaluate_predictions

@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str | Path | None = None

class SupervisedTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig):
        self.model, self.criterion, self.optimizer, self.config = model, criterion, optimizer, config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def _epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, total = 0.0, 0
        labels_all, preds_all, probs_all = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                if train: self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if train:
                    loss.backward(); self.optimizer.step()
                n = labels.size(0); total_loss += loss.item() * n; total += n
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(1)
                labels_all.extend(labels.detach().cpu().numpy()); preds_all.extend(preds.detach().cpu().numpy()); probs_all.append(probs.detach().cpu())
        metrics = evaluate_predictions(labels_all, preds_all, torch.cat(probs_all).numpy())
        metrics['loss'] = total_loss / total
        return metrics

    def fit(self, train_loader, val_loader):
        history, best = [], float('-inf')
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._epoch(train_loader, True)
            val_metrics = self._epoch(val_loader, False)
            row = {'epoch': epoch, **{f'train_{k}': v for k,v in train_metrics.items() if k != 'confusion_matrix'},
                   **{f'val_{k}': v for k,v in val_metrics.items() if k != 'confusion_matrix'}}
            history.append(row)
            if val_metrics['macro_f1'] > best:
                best = val_metrics['macro_f1']
                if self.config.checkpoint_path:
                    save_checkpoint(self.config.checkpoint_path, self.model, self.optimizer, epoch, val_metrics)
        return history

def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'state_dict': model.state_dict(), 'epoch': epoch, 'metrics': metrics or {}}
    if optimizer is not None: payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, path)
