"""Metric computation and threshold-selection tests."""
from __future__ import annotations

import numpy as np

from training.metrics import evaluate_predictions, select_threshold


def test_confusion_matrix_and_rates():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.6, 0.2, 0.9, 0.8, 0.4])
    m = evaluate_predictions(y_true, y_pred, scores)
    cm = m["confusion_matrix"]
    assert cm["true_negative"] == 2
    assert cm["false_positive"] == 1
    assert cm["false_negative"] == 1
    assert cm["true_positive"] == 2
    assert m["precision"] == round(2 / 3, 6)
    assert m["recall"] == round(2 / 3, 6)
    assert m["support"]["total"] == 6


def test_single_class_returns_none_for_ranking_metrics():
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    m = evaluate_predictions(y_true, y_pred, scores)
    assert m["pr_auc"] is None
    assert m["roc_auc"] is None
    assert m["notes"], "expected an explanatory note"
    assert m["false_positive_rate"] == 0.0


def test_threshold_selection_favours_recall():
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 500 + [1] * 50)
    scores = np.concatenate([rng.uniform(0, 0.6, 500), rng.uniform(0.4, 1.0, 50)])
    result = select_threshold(y_true, scores, beta=2.0)
    assert 0.0 < result["threshold"] < 1.0
    val = result["validation_metrics"]
    assert val["recall"] >= 0.5
    assert "F2" in result["objective"]


def test_threshold_selection_on_single_class_is_safe():
    y_true = np.zeros(20, dtype=int)
    scores = np.linspace(0, 1, 20)
    result = select_threshold(y_true, scores)
    assert isinstance(result["threshold"], float)


def test_perfect_separation_gives_perfect_scores():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.01, 0.02, 0.98, 0.99])
    m = evaluate_predictions(y_true, (scores >= 0.5).astype(int), scores)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
