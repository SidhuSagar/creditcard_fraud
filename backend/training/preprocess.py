"""FraudShield AI - training and inference share this module.

It is the single source of truth for:
  * the exact feature order used at training *and* inference time,
  * dataset loading / validation,
  * the reproducible train / validation / test split,
  * the (fit-on-train-only) preprocessing pipeline.

Nothing in this module may be changed without retraining the models, because
changing ``FEATURE_ORDER`` or the preprocessor would silently invalidate the
saved artifacts. ``tests/test_feature_order.py`` guards this contract.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Class"

V_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]  # V1 .. V28

# The exact order the model expects. Time first, then the 28 PCA components,
# then Amount. Verified by tests/test_feature_order.py.
FEATURE_ORDER: list[str] = ["Time", *V_COLUMNS, "Amount"]

# `Time` and `Amount` are raw seconds/dollars; V1..V28 are already PCA-scaled
# by the dataset authors, so we only scale the two raw columns.
SCALED_FEATURES: list[str] = ["Time", "Amount"]
PASSTHROUGH_FEATURES: list[str] = list(V_COLUMNS)

REQUIRED_COLUMNS: list[str] = [*FEATURE_ORDER, TARGET_COLUMN]

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20           # 20% held out, never touched until evaluation
VALIDATION_SIZE = 0.25     # 25% of the remaining 80% -> 20% of the whole set


class DatasetError(ValueError):
    """Raised when the dataset is missing, unreadable or malformed."""


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessing pipeline.

    The transformers are declared in ``FEATURE_ORDER`` order so the output
    column order is *identical* to ``FEATURE_ORDER``. We pass one transformer
    per group explicitly (instead of relying on ``remainder="passthrough"``)
    precisely so the output order cannot drift.
    """
    return ColumnTransformer(
        transformers=[
            ("time", StandardScaler(), ["Time"]),
            ("v_features", "passthrough", list(V_COLUMNS)),
            ("amount", StandardScaler(), ["Amount"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def assert_feature_order(preprocessor: ColumnTransformer) -> None:
    """Fail loudly if the fitted preprocessor emits columns out of order."""
    produced = list(preprocessor.get_feature_names_out())
    if produced != FEATURE_ORDER:
        raise AssertionError(
            "Preprocessor output order does not match FEATURE_ORDER.\n"
            f"  expected: {FEATURE_ORDER}\n"
            f"  produced: {produced}"
        )


# --------------------------------------------------------------------------- #
# Dataset loading / validation
# --------------------------------------------------------------------------- #
def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the raw dataframe and return a summary report."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            "Dataset is missing required column(s): "
            + ", ".join(missing)
            + ". Expected the Kaggle 'creditcard.csv' schema "
            "(Time, V1..V28, Amount, Class)."
        )

    non_numeric = [
        c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise DatasetError(
            "Dataset contains non-numeric feature column(s): "
            + ", ".join(non_numeric)
        )

    null_counts = {c: int(df[c].isnull().sum()) for c in REQUIRED_COLUMNS}
    null_counts = {c: n for c, n in null_counts.items() if n}
    if null_counts:
        raise DatasetError(f"Dataset contains null values: {null_counts}")

    classes = set(df[TARGET_COLUMN].unique().tolist())
    if not classes.issubset({0, 1}):
        raise DatasetError(
            f"'Class' must only contain 0/1, found: {sorted(classes)}"
        )

    n_total = int(len(df))
    n_fraud = int((df[TARGET_COLUMN] == 1).sum())
    n_legit = int((df[TARGET_COLUMN] == 0).sum())

    return {
        "rows": n_total,
        "fraud": n_fraud,
        "legitimate": n_legit,
        "fraud_rate": round(n_fraud / n_total, 8) if n_total else 0.0,
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "time_range_seconds": [
            float(df["Time"].min()),
            float(df["Time"].max()),
        ],
        "amount_stats": {
            "min": round(float(df["Amount"].min()), 4),
            "max": round(float(df["Amount"].max()), 4),
            "mean": round(float(df["Amount"].mean()), 4),
            "fraud_mean": round(
                float(df.loc[df[TARGET_COLUMN] == 1, "Amount"].mean()), 4
            ),
            "legit_mean": round(
                float(df.loc[df[TARGET_COLUMN] == 0, "Amount"].mean()), 4
            ),
        },
    }


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate ``creditcard.csv``. Raises :class:`DatasetError`."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place "
            "it in the data/ directory (see README.md)."
        )
    if path.stat().st_size == 0:
        raise DatasetError(f"Dataset at {path} is empty (0 bytes).")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - depends on file corruption
        raise DatasetError(f"Could not parse {path} as CSV: {exc}") from exc

    report = validate_dataset(df)
    report["path"] = str(path)
    report["size_bytes"] = int(path.stat().st_size)
    report["sha256"] = file_sha256(path)
    df = coerce_types(df)
    return df, report


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force the canonical dtypes so training and inference agree exactly."""
    out = df.copy()
    for col in FEATURE_ORDER:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = out[TARGET_COLUMN].astype("int64")
    return out


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    summary: dict[str, Any] = field(default_factory=dict)


def split_dataset(df: pd.DataFrame) -> DatasetSplit:
    """Stratified 60/20/20 train/validation/test split.

    The test set is held out first and never used for threshold selection or
    hyper-parameter tuning, which is what prevents evaluation leakage.
    """
    if TARGET_COLUMN not in df.columns:
        raise DatasetError("Cannot split: 'Class' column is missing.")

    X = df[FEATURE_ORDER]
    y = df[TARGET_COLUMN]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=VALIDATION_SIZE,
        stratify=y_rest,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    def _sta(yy: pd.Series) -> dict[str, int]:
        return {"rows": int(len(yy)), "fraud": int(yy.sum())}

    summary = {
        "strategy": "stratified random split, seed=42",
        "train": _sta(y_train),
        "validation": _sta(y_val),
        "test": _sta(y_test),
    }

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# JSON helper
# --------------------------------------------------------------------------- #
def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty UTF-8 JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
        fh.write("\n")


def _json_default(obj: Any):  # pragma: no cover - defensive
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":  # manual smoke check
    import sys

    default = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    _df, _report = load_dataset(target)
    print(json.dumps(_report, indent=2)[:2000])
