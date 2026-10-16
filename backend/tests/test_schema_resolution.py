"""Unit tests for flexible column-name resolution.

The model is fitted on the Kaggle ``creditcard.csv`` names, but an upload may
name its columns anything. These tests pin the fuzzy-matching contract in
``analyze_service.resolve_columns``.
"""
from __future__ import annotations

from app.services.analyze_service import _normalise, resolve_columns
from training.preprocess import FEATURE_ORDER, TARGET_COLUMN, V_COLUMNS


def test_canonical_names_resolve_exactly():
    rename, missing = resolve_columns([*FEATURE_ORDER, TARGET_COLUMN])
    assert missing == []
    assert rename["Time"] == "Time"
    assert rename["Amount"] == "Amount"
    assert rename["V1"] == "V1"
    assert rename["V28"] == "V28"
    assert rename[TARGET_COLUMN] == TARGET_COLUMN


def test_normalise_is_forgiving():
    assert _normalise("\ufeff Time ") == "time"
    assert _normalise("Transaction Dollar Amount") == "transactiondollaramount"
    assert _normalise("v01") == "v01"


def test_common_aliases_are_recognised():
    columns = [
        "\ufeffTime", "V01", "v2", "Transaction_Dollar_Amount",
        *[f"v{i}" for i in range(3, 29)],
        "is_fraud",
    ]
    rename, missing = resolve_columns(columns)
    assert missing == []
    assert rename["\ufeffTime"] == "Time"
    assert rename["V01"] == "V1"
    assert rename["v2"] == "V2"
    assert rename["Transaction_Dollar_Amount"] == "Amount"
    assert rename["is_fraud"] == TARGET_COLUMN


def test_unrelated_columns_are_reported_missing():
    # The synthetic 'transactions.csv' shape: no V-features, no labels.
    rename, missing = resolve_columns(
        ["credit_card", "date", "transaction_dollar_amount", "Long", "Lat"]
    )
    # Amount is recognised, everything else is not.
    assert rename == {"transaction_dollar_amount": "Amount"}
    assert set(missing) == set(FEATURE_ORDER) - {"Amount"}
    assert "V1" in missing and "Time" in missing


def test_duplicate_normalised_column_is_not_reused():
    # Two columns that normalise to the same key must not both be consumed.
    rename, missing = resolve_columns([*FEATURE_ORDER, " Time "])
    assert missing == []
    assert len(rename) == len(FEATURE_ORDER)
    assert rename["Time"] == "Time"
    assert " Time " not in rename


def test_label_aliases():
    for alias in ("class", "is_fraud", "label", "target", "fraudulent"):
        columns = [*FEATURE_ORDER, alias]
        rename, missing = resolve_columns(columns)
        assert missing == []
        assert rename[alias] == TARGET_COLUMN


def test_v_column_count_matches_feature_order():
    assert len(V_COLUMNS) == 28
