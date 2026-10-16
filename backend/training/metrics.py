"""Metric computation and validation-set threshold selection.

Shared by ``train_models.py`` and ``evaluate_models.py`` so that a metric is
computed in exactly one place.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Recall-weighted F-beta. beta=2 makes recall ~twice as important as
# precision, which matches the project objective ("prioritise fraud recall
# while monitoring precision and false positives").
DEFAULT_BETA = 2.0


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Return the full metric bundle for one model on one dataset.

    Ranking metrics (PR-AUC / ROC-AUC) are only defined when both classes are
    present. When the evaluated slice contains a single class they are returned
    as ``None`` together with an explanatory note - never silently as 0, which
    would be misleading.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) >= 2
    notes: list[str] = []
    if both_classes:
        pr_auc: float | None = round(float(average_precision_score(y_true, y_score)), 6)
        roc_auc: float | None = round(float(roc_auc_score(y_true, y_score)), 6)
    else:
        pr_auc = None
        roc_auc = None
        notes.append(
            "PR-AUC and ROC-AUC are undefined: the evaluated rows contain only "
            f"one class (all {int(y_true[0]) if len(y_true) else 'n/a'})."
        )

    metrics: dict[str, Any] = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": round(float((tp + tn) / max(len(y_true), 1)), 6),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_rate": round(_safe_div(fp, fp + tn), 6),
        "false_negative_rate": round(_safe_div(fn, fn + tp), 6),
        "support": {
            "total": int(len(y_true)),
            "actual_fraud": int((y_true == 1).sum()),
            "actual_legitimate": int((y_true == 0).sum()),
            "predicted_fraud": int((y_pred == 1).sum()),
        },
        "notes": notes,
    }
    return metrics


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = DEFAULT_BETA,
) -> dict[str, Any]:
    """Pick a decision threshold on the *validation* set using F-beta.

    Higher ``y_score`` must mean "more likely fraud" (for Isolation Forest we
    pass the negated anomaly score). Returns the threshold plus the metrics
    observed at that threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        # Degenerate validation split - fall back to the median score.
        thr = float(np.median(y_score)) if len(y_score) else 0.5
        pred = (y_score >= thr).astype(int)
        return {
            "threshold": thr,
            "beta": beta,
            "objective": f"F{beta:g}",
            "validation_metrics": evaluate_predictions(y_true, pred, y_score),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall are one element longer than thresholds
    p = precision[:-1]
    r = recall[:-1]
    beta_sq = beta * beta
    denom = beta_sq * p + r
    fbeta = np.divide(
        (1 + beta_sq) * p * r,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    best_idx = int(np.argmax(fbeta))
    thr = float(thresholds[best_idx]) if len(thresholds) else 0.5

    # Guard against the pathological "flag everything" threshold.
    if not np.isfinite(thr):
        thr = float(np.median(y_score))

    pred = (y_score >= thr).astype(int)
    return {
        "threshold": round(thr, 8),
        "beta": beta,
        "objective": f"F{beta:g} (recall-weighted) maximised on validation set",
        "validation_metrics": evaluate_predictions(y_true, pred, y_score),
    }
