"""Train the generic amount/date anomaly detector.

This is an **unsupervised** detector trained on the generic transaction dataset
``transactions.csv`` (at the project root). It learns what looks "normal"
using only ``transaction_dollar_amount`` and the time-of-day / day-of-week
derived from a ``date`` column. It is used only as a fallback when a file does
not contain the 30 creditcard.csv features.

No ground-truth labels are used, so it produces no accuracy/precision/recall
figures - the output is labelled as "Anomaly" / "Normal" and is clearly
marked as unvalidated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
GENERIC_DATA = PROJECT_ROOT / "transactions.csv"
OUT_JOBLIB = ARTIFACTS_DIR / "generic_anomaly.joblib"
OUT_META = ARTIFACTS_DIR / "generic_metadata.json"

RANDOM_STATE = 42


def _normalise_date(s: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        dt = pd.to_datetime(s, errors="coerce")
    return dt


def load_generic(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Generic dataset not found at {path}")
    df = pd.read_csv(path)
    # find amount
    amount_col = None
    for c in df.columns:
        cc = str(c).strip().lower().replace(" ", "_")
        if cc in ("transaction_dollar_amount", "amount", "amt"):
            amount_col = c
            break
    if amount_col is None:
        raise ValueError(f"No amount column found in {path}; looked for 'transaction_dollar_amount'/'amount'/'amt'")

    # find date
    dt_col = None
    for c in df.columns:
        cc = str(c).strip().lower().replace(" ", "_")
        if cc in ("date", "datetime", "transaction_date", "transactiondatetime"):
            dt_col = c
            break

    dt = _normalise_date(df[dt_col]) if dt_col else pd.Series(pd.NaT, index=df.index)

    X = pd.DataFrame({
        "log_amount": np.log1p(df[amount_col].astype(float).clip(lower=0)),
    })
    X["hour"] = dt.dt.hour if dt is not None else 0
    X["dow"] = dt.dt.dayofweek if dt is not None else 0
    X["hour"] = X["hour"].fillna(0).astype(float)
    X["dow"] = X["dow"].fillna(0).astype(float)

    meta = {
        "source": str(path),
        "rows_total": int(len(df)),
        "amount_col": amount_col,
        "datetime_col": dt_col,
        "features": ["log_amount", "hour", "dow"],
    }
    return X, meta


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X, meta = load_generic(GENERIC_DATA)
    # train on all rows (unsupervised)
    est = IsolationForest(n_estimators=200, contamination="auto", max_samples="auto", n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    est.fit(X)
    artifacts = {
        "estimator": est,
        "features": meta["features"],
        "impute": {"hour": float(0), "dow": float(0)},
        "contamination": "auto",
        "trained_on": meta["source"],
        "trained_on_rows": int(X.shape[0]),
        "random_state": RANDOM_STATE,
        "version": "1.0.0",
    }
    joblib.dump(artifacts, OUT_JOBLIB)
    meta_out = {
        "trained_on": meta["source"],
        "rows": int(X.shape[0]),
        "features": meta["features"],
        "impute": artifacts["impute"],
        "contamination": "auto",
        "random_state": RANDOM_STATE,
        "version": "1.0.0",
        "note": "Unsupervised anomaly detector for amount/date; not a validated fraud model.",
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Saved: {OUT_JOBLIB} (rows={X.shape[0]})")


if __name__ == "__main__":
    main()
