"""Tests for the feature-order / leakage contract in training.preprocess."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.preprocess import (
    FEATURE_ORDER,
    DatasetError,
    build_preprocessor,
    coerce_types,
    split_dataset,
    validate_dataset,
)


def _mini_frame(n: int = 200, fraud_ratio: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_fraud = max(2, int(n * fraud_ratio))
    data = {"Time": rng.uniform(0, 100000, n), "Amount": rng.uniform(1, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    frame = pd.DataFrame(data)
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    frame["Class"] = labels
    return frame


def test_feature_order_is_exactly_30_features():
    assert len(FEATURE_ORDER) == 30
    assert FEATURE_ORDER[0] == "Time"
    assert FEATURE_ORDER[-1] == "Amount"
    assert FEATURE_ORDER[1:29] == [f"V{i}" for i in range(1, 29)]


def test_preprocessor_output_order_matches_feature_order():
    frame = _mini_frame()
    pre = build_preprocessor()
    out = pre.fit_transform(frame[FEATURE_ORDER])
    assert out.shape[1] == 30
    assert list(pre.get_feature_names_out()) == FEATURE_ORDER


def test_preprocessor_transforms_dataframe_input_at_inference_time():
    """Regression guard: fitting on a DataFrame requires a DataFrame to transform."""
    frame = _mini_frame()
    pre = build_preprocessor().fit(frame[FEATURE_ORDER])
    one_row = frame[FEATURE_ORDER].iloc[[0]]
    assert pre.transform(one_row).shape == (1, 30)
    with pytest.raises(Exception):
        pre.transform(np.asarray(one_row))  # numpy is not acceptable here


def test_only_time_and_amount_are_scaled():
    frame = _mini_frame(400)
    pre = build_preprocessor().fit(frame[FEATURE_ORDER])
    out = pre.transform(frame[FEATURE_ORDER])
    time_idx = FEATURE_ORDER.index("Time")
    amount_idx = FEATURE_ORDER.index("Amount")
    assert abs(float(out[:, time_idx].mean())) < 1e-9
    assert abs(float(out[:, time_idx].std()) - 1.0) < 1e-6
    assert abs(float(out[:, amount_idx].mean())) < 1e-9
    # V columns must be untouched
    v_idx = FEATURE_ORDER.index("V1")
    np.testing.assert_allclose(out[:, v_idx], frame["V1"].to_numpy())


def test_scaler_is_fit_on_train_only():
    """The scaler must learn TRAIN statistics, not whole-dataset statistics.

    This is the concrete guard against preprocessing leakage: if the scaler had
    seen the validation/test rows its learned mean would drift towards the
    overall dataset mean.
    """
    frame = _mini_frame(500)
    split = split_dataset(frame)
    pre = build_preprocessor().fit(split.X_train)

    time_scaler = pre.named_transformers_["time"]
    amount_scaler = pre.named_transformers_["amount"]

    assert float(time_scaler.mean_[0]) == pytest.approx(
        float(split.X_train["Time"].mean()), abs=1e-9
    )
    assert float(amount_scaler.mean_[0]) == pytest.approx(
        float(split.X_train["Amount"].mean()), abs=1e-9
    )
    # And explicitly NOT the whole-dataset statistics.
    assert abs(float(time_scaler.mean_[0]) - float(frame["Time"].mean())) > 1e-9

    out = pre.transform(split.X_train[FEATURE_ORDER])
    assert abs(float(out[:, 0].mean())) < 1e-9
    assert abs(float(out[:, 0].std()) - 1.0) < 1e-6


def test_split_is_stratified_and_disjoint():
    frame = _mini_frame(1000, fraud_ratio=0.1)
    split = split_dataset(frame)
    total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    assert total == len(frame)
    assert split.summary["train"]["rows"] == len(split.y_train)
    # each split should contain frauds
    assert split.y_train.sum() > 0 and split.y_val.sum() > 0 and split.y_test.sum() > 0


def test_validate_dataset_rejects_missing_columns():
    frame = _mini_frame().drop(columns=["V12"])
    with pytest.raises(DatasetError, match="V12"):
        validate_dataset(frame)


def test_validate_dataset_rejects_bad_class_values():
    frame = _mini_frame()
    frame.loc[0, "Class"] = 7
    with pytest.raises(DatasetError, match="0/1"):
        validate_dataset(frame)


def test_validate_dataset_rejects_nulls():
    frame = _mini_frame()
    frame.loc[0, "V5"] = np.nan
    with pytest.raises(DatasetError, match="null"):
        validate_dataset(frame)


def test_validate_dataset_accepts_good_frame():
    report = validate_dataset(_mini_frame())
    assert report["rows"] == 200
    assert report["fraud"] + report["legitimate"] == 200


def test_coerce_types_uses_float64():
    frame = _mini_frame().astype({c: "float32" for c in FEATURE_ORDER})
    out = coerce_types(frame)
    for col in FEATURE_ORDER:
        assert out[col].dtype == "float64"
    assert out["Class"].dtype == "int64"
