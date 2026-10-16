from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.services import generic_detector
from app.services.generic_detector import GenericNotAvailableError
from app.services.model_registry import registry
from training.metrics import evaluate_predictions
from training.preprocess import FEATURE_ORDER, TARGET_COLUMN
from training.train_models import MODEL_ORDER
from app.services import schema

# Re-export for backwards compatibility with tests.
_normalise = schema.normalise
resolve_columns = schema.resolve_columns
found_columns = schema.found_columns

logger = logging.getLogger("fraudshield.analyze")

MAX_REJECTED_REPORTED = 200


class UploadError(ValueError):
    pass


def _row_number(position: int) -> int:
    return int(position) + 2


def _analyze_supervised(frame, filename, model_key, rename):
    work = frame.rename(columns=rename)
    total_rows = int(len(work))
    numeric = work.loc[:, FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.notna().all(axis=1).to_numpy()
    valid_positions = np.flatnonzero(valid_mask)
    invalid_positions = np.flatnonzero(~valid_mask)
    if valid_positions.size == 0:
        raise UploadError("None of the rows contained valid numeric values for the required factors (Time, V1-V28, Amount).")
    if valid_positions.size > settings.max_batch_rows:
        raise UploadError(f"The file has {valid_positions.size:,} usable rows, exceeding the {settings.max_batch_rows:,}-row limit.")
    usable = numeric.iloc[valid_positions]
    scored = registry.predict_dataframe(model_key, usable)
    predictions = np.array([r["prediction"] for r in scored], dtype=int)
    scores = np.array([r["score"] for r in scored], dtype=float)
    amounts = usable["Amount"].to_numpy(dtype=float)
    actual_values = None
    if TARGET_COLUMN in work.columns:
        actual_series = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
        candidate = actual_series.iloc[valid_positions]
        if candidate.notna().all():
            actual_values = candidate.astype(int).to_numpy()
    order = np.lexsort((-scores, -predictions))
    shown = order[: settings.max_response_rows]
    results_truncated = bool(shown.size < predictions.size)
    results = [{
        "row": _row_number(int(valid_positions[i])),
        "amount": float(amounts[i]),
        "prediction": int(predictions[i]),
        "label": "Fraud" if predictions[i] == 1 else "Legitimate",
        "score": float(scores[i]),
        "actual_class": int(actual_values[i]) if actual_values is not None else None,
    } for i in shown]
    rejected = [{
        "row": _row_number(int(pos)),
        "reason": "missing or non-numeric value in a required feature column",
    } for pos in invalid_positions[:MAX_REJECTED_REPORTED]]
    evaluation = evaluate_predictions(actual_values, predictions, scores) if actual_values is not None else None
    fraud_rows = int((predictions == 1).sum())
    processed_rows = int(predictions.size)
    return {
        "filename": filename,
        "model": model_key,
        "model_display_name": scored[0]["model_display_name"] if scored else model_key,
        "score_type": scored[0]["score_type"] if scored else "probability",
        "threshold": scored[0]["threshold"] if scored else registry.get(model_key).threshold,
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "rejected_rows": int(invalid_positions.size),
        "fraud_rows": fraud_rows,
        "legitimate_rows": processed_rows - fraud_rows,
        "results": results,
        "results_truncated": results_truncated,
        "rejected": rejected,
        "rejected_truncated": int(invalid_positions.size) > MAX_REJECTED_REPORTED,
        "evaluation": evaluation,
        "mode": "supervised",
    }


def _analyze_generic(frame, filename, amount_col, datetime_col=None):
    results, meta = generic_detector.score_frame(frame, amount_col, datetime_col)
    work_frame = frame.reset_index(drop=True)
    rows_out = []
    anomaly = 0
    processed = len(results)
    for i, r in enumerate(results):
        pos = i + 2
        pred = r["prediction"]
        anomaly += pred
        amount_val = float(work_frame[amount_col].iloc[i]) if i < len(work_frame) else r["amount"]
        if pd.isna(amount_val):
            amount_val = r["amount"]
        rows_out.append({
            "row": pos,
            "amount": float(amount_val),
            "prediction": pred,
            "label": "Anomaly" if pred == 1 else "Normal",
            "score": r["score"],
            "actual_class": None,
        })
    rows_out.sort(key=lambda x: (-x["prediction"], -x["score"]))
    shown = rows_out[: settings.max_response_rows]
    return {
        "filename": filename,
        "model": meta["model"],
        "model_display_name": meta["model_display_name"],
        "mode": "generic",
        "mode_label": meta["mode_label"],
        "mode_note": meta["mode_note"],
        "score_type": meta["score_type"],
        "threshold": meta["threshold"],
        "total_rows": int(len(frame)),
        "processed_rows": processed,
        "rejected_rows": 0,
        "fraud_rows": anomaly,
        "legitimate_rows": processed - anomaly,
        "results": shown,
        "results_truncated": len(shown) < processed,
        "rejected": [],
        "rejected_truncated": False,
        "evaluation": None,
    }


def analyze_frame(frame, filename, model_key):
    if model_key not in MODEL_ORDER:
        raise UploadError(f"Unknown model '{model_key}'. Available: {', '.join(MODEL_ORDER)}.")
    try:
        registry.get(model_key)
    except Exception as exc:
        from app.services.model_registry import ModelNotAvailableError
        if isinstance(exc, ModelNotAvailableError):
            raise exc

    if frame.empty:
        raise UploadError("The uploaded CSV has no data rows.")
    rename, missing = schema.resolve_columns(frame.columns)
    if missing:
        amount_col = schema.find_amount_col(frame.columns)
        if amount_col is not None and generic_detector.available():
            dt_col = schema.find_datetime_col(frame.columns)
            return _analyze_generic(frame, filename, amount_col, dt_col)
        raise UploadError(
            "This file does not contain the transaction factors the model was trained on. "
            f"Missing: {', '.join(missing)}. Found columns: {schema.found_columns(frame.columns)}. "
            "The model expects the Kaggle creditcard.csv factors - Time, V1-V28 and Amount "
            "(or for non-ULB files, an amount column to use the generic anomaly mode)."
        )
    return _analyze_supervised(frame, filename, model_key, rename)


def analyze_csv(content: bytes, filename: str, model_key: str):
    if not content:
        raise UploadError("The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise UploadError(f"Could not read the CSV file: {exc}") from exc
    return analyze_frame(frame, filename, model_key)


def available_models():
    return registry.catalogue()
