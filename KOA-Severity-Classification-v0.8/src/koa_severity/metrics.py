"""Ordinal-aware evaluation metrics used across the research notebooks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    mean_absolute_error, roc_auc_score,
)


def ordinal_class_probabilities(
    threshold_probabilities: np.ndarray,
    eps: float = 1e-7,
    strict: bool = False,
) -> np.ndarray:
    """Convert P(Y>k) for k=0..K-2 into K class probabilities.

    In an ordinal model the threshold probabilities should be non-increasing.
    With ``strict=True`` invalid rows raise ``ValueError`` instead of being
    silently repaired by clipping/normalization.
    """
    q = np.asarray(threshold_probabilities, dtype=float)
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError("Expected a 2-D array with K-1 threshold probabilities.")
    if not np.isfinite(q).all():
        raise ValueError("Threshold probabilities contain non-finite values.")
    if ((q < 0) | (q > 1)).any():
        raise ValueError("Threshold probabilities must lie in [0, 1].")
    if strict and np.any(np.diff(q, axis=1) > eps):
        raise ValueError("Ordinal threshold probabilities must be non-increasing.")

    n, k_minus_1 = q.shape
    probs = np.zeros((n, k_minus_1 + 1), dtype=float)
    probs[:, 0] = 1.0 - q[:, 0]
    if k_minus_1 > 1:
        probs[:, 1:-1] = q[:, :-1] - q[:, 1:]
    probs[:, -1] = q[:, -1]
    probs = np.clip(probs, eps, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _macro_roc_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float | None:
    """Compute multiclass OVR macro AUC, returning None when undefined."""
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"class_probabilities must have shape (n, {num_classes}).")
    present = np.unique(y_true)
    if present.size < 2:
        return None
    if present.size != num_classes:
        # sklearn cannot define a standard full multiclass AUC when classes are absent.
        return None
    try:
        return float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro",
            labels=np.arange(num_classes),
        ))
    except ValueError:
        return None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_probabilities: np.ndarray | None = None,
    num_classes: int = 5,
) -> dict:
    """Return standard and ordinal-aware metrics with explicit edge handling."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 1:
        y_true, y_pred = y_true.ravel(), y_pred.ravel()
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "adjacent_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
    }
    if class_probabilities is not None:
        auc = _macro_roc_auc(y_true, np.asarray(class_probabilities, dtype=float), num_classes)
        if auc is not None:
            result["macro_roc_auc_ovr"] = auc
    return result
