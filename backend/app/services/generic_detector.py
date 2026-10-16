from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.services.schema import normalise as _normalise

logger = logging.getLogger("fraudshield.generic")

AMOUNT_ALIASES = (
    "amount", "amt", "amountusd", "amount_usd", "transactionamount",
    "transaction_amount", "transactionamountusd", "transactiondollaramount",
    "transaction_dollar_amount", "dollaramount", "value", "transactionvalue",
    "money", "price",
)

DATETIME_ALIASES = (
    "date", "datetime", "transactiondate", "transactiondatetime",
    "time", "timestamp", "timestamps", "unixtime", "unix_timestamp",
    "transactiontime",
)


def _find_amount_col(cols):
    for c in cols:
        if _normalise(c) in {_normalise(n) for n in AMOUNT_ALIASES}:
            return c
    return None


def _find_datetime_col(cols):
    for c in cols:
        if _normalise(c) in {_normalise(n) for n in DATETIME_ALIASES}:
            return c
    return None


class GenericNotAvailableError(RuntimeError):
    pass


class GenericDetector:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.artifacts_dir / "generic_anomaly.joblib")
        self._loaded = False
        self._artifacts = None

    def load(self):
        if self._loaded:
            return
        if not self.path.exists():
            raise GenericNotAvailableError(f"Generic detector not found at {self.path}")
        self._artifacts = joblib.load(self.path)
        self._loaded = True

    @property
    def available(self):
        return self.path.exists()

    def score_frame(self, frame: pd.DataFrame, amount_col: str, datetime_col: str | None = None) -> tuple[list[dict], dict]:
        self.load()
        a = joblib.load(self.path) if False else self._artifacts  # dummy
        a = self._artifacts
        est = a["estimator"]
        features = a["features"]
        impute = a.get("impute", {"hour": 0.0, "dow": 0.0})
        X = pd.DataFrame(index=frame.index)
        X["log_amount"] = np.log1p(pd.to_numeric(frame[amount_col], errors="coerce").astype(float).clip(lower=0))
        if datetime_col is not None:
            dt = pd.to_datetime(frame[datetime_col], errors="coerce")
            X["hour"] = dt.dt.hour.fillna(impute["hour"]).astype(float)
            X["dow"] = dt.dt.dayofweek.fillna(impute["dow"]).astype(float)
        else:
            X["hour"] = float(impute["hour"])
            X["dow"] = float(impute["dow"])
        X = X.reindex(columns=features, fill_value=0.0)
        scores = -est.score_samples(X)
        offset = getattr(est, "offset_", 0.0)
        threshold = float(-offset) if hasattr(est, "offset_") else float(np.percentile(scores, 95))  # fallback
        results = []
        for i in range(len(X)):
            s = float(scores[i])
            pred = 1 if s >= threshold else 0
            results.append({
                "row": i + 2,  # will be overridden by caller
                "amount": float(frame[amount_col].iloc[i]) if pd.notna(frame[amount_col].iloc[i]) else float(X["log_amount"].iloc[i]),
                "prediction": pred,
                "label": "Anomaly" if pred == 1 else "Normal",
                "score": round(s, 6),
            })
        meta = {
            "model": "generic_anomaly",
            "model_display_name": "Generic amount/date anomaly detector",
            "mode": "generic",
            "mode_label": "Unsupervised anomaly detection (not the trained fraud model)",
            "mode_note": "No fraud labels available; scores are relative to this file.",
            "score_type": "anomaly_score",
            "threshold": threshold,
            "features": features,
            "trained_on_rows": a.get("trained_on_rows"),
        }
        return results, meta


_det = GenericDetector()


def available():
    return _det.available


def score_frame(frame, amount_col, datetime_col=None):
    return _det.score_frame(frame, amount_col, datetime_col)
