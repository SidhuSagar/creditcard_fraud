"""Shared pytest fixtures."""
from __future__ import annotations

import io
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import model_registry  # noqa: E402
from training.preprocess import FEATURE_ORDER, TARGET_COLUMN  # noqa: E402

ARTIFACTS_DIR = settings.artifacts_dir
MODELS_READY = model_registry.registry.models_trained
DATASET_READY = settings.dataset_available

needs_models = pytest.mark.skipif(
    not MODELS_READY,
    reason="Trained model artifacts are not present; run training/train_models.py.",
)
needs_dataset = pytest.mark.skipif(
    not DATASET_READY,
    reason="creditcard.csv is not present; see README.md for dataset setup.",
)

needs_generic = pytest.mark.skipif(
    False,
    reason="Generic anomaly detector not trained; run training/train_generic.py.",
)


# Mean of the 492 genuine fraud rows in creditcard.csv ("the fraud centroid").
FRAUD_CENTROID: dict[str, float] = {
    "Time": 80746.8069, "V1": -4.7719, "V2": 3.6238, "V3": -7.0333,
    "V4": 4.5420, "V5": -3.1512, "V6": -1.3977, "V7": -5.5687,
    "V8": 0.5706, "V9": -2.5811, "V10": -5.6769, "V11": 3.8002,
    "V12": -6.2594, "V13": -0.1093, "V14": -6.9717, "V15": -0.0929,
    "V16": -4.1399, "V17": -6.6658, "V18": -2.2463, "V19": 0.6807,
    "V20": 0.3723, "V21": 0.7136, "V22": 0.0140, "V23": -0.0403,
    "V24": -0.1051, "V25": 0.0414, "V26": 0.0516, "V27": 0.1706,
    "V28": 0.0757, "Amount": 122.2113,
}


def fraud_like_transaction(**overrides) -> dict:
    payload = dict(FRAUD_CENTROID)
    payload.update(overrides)
    return payload


def legit_like_transaction(**overrides) -> dict:
    payload = {"Time": 406.0, "Amount": 1.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    payload.update(overrides)
    return payload


def transactions_to_csv(rows: list[dict], labels: list[int] | None = None) -> bytes:
    frame = pd.DataFrame(rows, columns=FEATURE_ORDER)
    if labels is not None:
        frame[TARGET_COLUMN] = labels
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
