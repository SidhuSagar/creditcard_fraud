"""POST /api/analyze and the single-page app's contract."""
from __future__ import annotations

import io

import pandas as pd

from app.services.model_registry import ModelRegistry
from training.preprocess import FEATURE_ORDER

from tests.conftest import (
    fraud_like_transaction,
    legit_like_transaction,
    needs_models,
    needs_generic,
    transactions_to_csv,
)