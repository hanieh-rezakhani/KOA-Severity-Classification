import numpy as np
import pytest
from koa_severity.metrics import evaluate_predictions, ordinal_class_probabilities

def test_ordinal_probabilities_sum_to_one():
    p = ordinal_class_probabilities(np.array([[0.8, 0.4, 0.2, 0.1]]), strict=True)
    assert p.shape == (1, 5) and np.allclose(p.sum(axis=1), 1.0)

def test_ordinal_probabilities_reject_non_monotonic_in_strict_mode():
    with pytest.raises(ValueError):
        ordinal_class_probabilities(np.array([[0.2, 0.8, 0.1, 0.7]]), strict=True)

def test_evaluate_predictions_contains_ordinal_metrics():
    y = np.array([0, 1, 2, 3, 4])
    result = evaluate_predictions(y, y)
    assert result["accuracy"] == 1.0 and result["macro_f1"] == 1.0
    assert result["qwk"] == 1.0 and result["mae"] == 0.0
    assert result["adjacent_accuracy"] == 1.0

def test_evaluate_predictions_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        evaluate_predictions(np.array([0, 1]), np.array([0]))
