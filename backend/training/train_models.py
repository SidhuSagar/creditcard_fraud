"""FraudShield AI - model training entry point.

Trains and compares three models on the ULB Credit Card Fraud dataset:

  * ``random_forest``    - supervised, class-imbalance handled by SMOTE applied
                           to the training fold ONLY.
  * ``xgboost``          - supervised gradient boosting, imbalance handled with
                           ``scale_pos_weight`` (no resampling).
  * ``isolation_forest`` - unsupervised anomaly detection; its raw anomaly
                           score is thresholded on the validation split.

Discipline that makes the reported numbers trustworthy:

  1. The 20% test split is created once, up front, and is never used for
     training, early stopping, or threshold selection.
  2. Preprocessing (standard scaling) is fitted on the training split only.
  3. SMOTE is applied to the training split only.
  4. Decision thresholds are chosen on the validation split with a
     recall-weighted F2 objective.
  5. The "recommended" model is chosen by validation F2, *not* by test score.

Usage:
    python backend/training/train_models.py
    python backend/training/train_models.py --data data/creditcard.csv --quick
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

# Make ``backend`` importable when run as a script from the project root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.metrics import evaluate_predictions, select_threshold  # noqa: E402
from training.preprocess import (  # noqa: E402
    FEATURE_ORDER,
    RANDOM_STATE,
    DatasetError,
    assert_feature_order,
    build_preprocessor,
    load_dataset,
    split_dataset,
    write_json,
)

PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "creditcard.csv"
DEFAULT_ARTIFACTS = BACKEND_DIR / "artifacts"

MODEL_ORDER = ["random_forest", "xgboost", "isolation_forest"]

MODEL_INFO: dict[str, dict[str, str]] = {
    "random_forest": {
        "display_name": "Random Forest",
        "family": "supervised",
        "score_type": "probability",
        "description": (
            "Ensemble of decision trees (bagging). Class imbalance is handled "
            "with SMOTE oversampling of the training fold only."
        ),
    },
    "xgboost": {
        "display_name": "XGBoost",
        "family": "supervised",
        "score_type": "probability",
        "description": (
            "Gradient-boosted decision trees with early stopping on the "
            "validation set; imbalance handled with scale_pos_weight."
        ),
    },
    "isolation_forest": {
        "display_name": "Isolation Forest",
        "family": "unsupervised",
        "score_type": "anomaly_score",
        "description": (
            "Unsupervised anomaly detector. Produces an anomaly score, not a "
            "calibrated probability; the score is thresholded on validation data."
        ),
    },
}


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_random_forest() -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,          # imbalance handled by SMOTE instead
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_xgboost(scale_pos_weight: float) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=40,
    )


def build_isolation_forest() -> Any:
    from sklearn.ensemble import IsolationForest

    return IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
# Score extraction helpers
# --------------------------------------------------------------------------- #
def rf_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Fraud probability from predict_proba (column 1)."""
    return model.predict_proba(X)[:, 1]


def xgb_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Fraud probability from XGBoost (column 1)."""
    return model.predict_proba(X)[:, 1]


def iforest_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Anomaly score where HIGHER means MORE anomalous.

    ``score_samples`` returns the negative of the anomaly score, so we negate
    it. This is *not* a probability and is not interpreted as one.
    """
    return -model.score_samples(X)


SCORE_FUNCS: dict[str, Callable[[Any, np.ndarray], np.ndarray]] = {
    "random_forest": rf_score,
    "xgboost": xgb_score,
    "isolation_forest": iforest_score,
}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_all(
    data_path: Path,
    artifacts_dir: Path,
    quick: bool = False,
    only: list[str] | None = None,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    _log(f"loading dataset from {data_path}")
    df, dataset_report = load_dataset(data_path)
    _log(
        f"dataset ok: {dataset_report['rows']:,} rows, "
        f"{dataset_report['fraud']} fraud "
        f"({dataset_report['fraud_rate'] * 100:.4f}%)"
    )

    if quick:
        # Small, still-stratified subsample so the pipeline can be smoked
        # tested in seconds. Never used for the numbers in the README.
        frauds = df[df["Class"] == 1]
        legit = df[df["Class"] == 0].sample(
            n=min(4000, (df["Class"] == 0).sum()), random_state=RANDOM_STATE
        )
        df = (
            __import__("pandas")
            .concat([legit, frauds])
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        _log(f"QUICK mode: subsampled to {len(df):,} rows")

    split = split_dataset(df)
    _log(f"split: {split.summary}")

    # ---- preprocessing: fit on TRAIN only -------------------------------- #
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)
    assert_feature_order(preprocessor)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = split.y_train.to_numpy()
    y_val = split.y_val.to_numpy()
    y_test = split.y_test.to_numpy()

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    _log("saved preprocessor.joblib")

    models_to_train = only or MODEL_ORDER
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "dataset": {
            k: v
            for k, v in dataset_report.items()
            if k not in {"columns"}
        },
        "dataset_columns": dataset_report["columns"],
        "splits": split.summary,
        "feature_order": FEATURE_ORDER,
        "preprocessing": {
            "scaled_features": ["Time", "Amount"],
            "passthrough_features": [c for c in FEATURE_ORDER if c not in ("Time", "Amount")],
            "fitted_on": "training split only",
            "resampling": "SMOTE (training fold only) for Random Forest; "
                          "scale_pos_weight for XGBoost; none for Isolation Forest",
        },
        "selection_objective": (
            "Validation-set F2 (beta=2, recall-weighted). Test metrics are "
            "reported for the chosen threshold and are never used to select "
            "the threshold or the recommended model."
        ),
        "models": {},
    }

    # ---------------- Random Forest (SMOTE on train fold) ----------------- #
    if "random_forest" in models_to_train:
        from imblearn.over_sampling import SMOTE

        _log("training random_forest (SMOTE on training fold)")
        t0 = time.time()
        # NB: imbalanced-learn 0.12 deprecated SMOTE's own n_jobs parameter;
        # the NearestNeighbors estimator it builds is not parallel here, so
        # we simply omit it.
        smote = SMOTE(
            sampling_strategy=0.1,
            k_neighbors=5,
            random_state=RANDOM_STATE,
        )
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        _log(
            f"  SMOTE: {len(y_train):,} -> {len(y_train_sm):,} rows "
            f"({int(y_train_sm.sum())} fraud)"
        )
        rf = build_random_forest()
        rf.fit(X_train_sm, y_train_sm)
        score_fn = SCORE_FUNCS["random_forest"]
        val_scores = score_fn(rf, X_val)
        test_scores = score_fn(rf, X_test)
        thr_info = select_threshold(y_val, val_scores)
        thr = thr_info["threshold"]
        test_metrics = evaluate_predictions(
            y_test, (test_scores >= thr).astype(int), test_scores
        )
        joblib.dump(rf, artifacts_dir / "random_forest.joblib")
        metadata["models"]["random_forest"] = {
            **MODEL_INFO["random_forest"],
            "artifact": "random_forest.joblib",
            "trained": True,
            "hyperparameters": rf.get_params(),
            "resampling": {
                "method": "SMOTE",
                "sampling_strategy": 0.1,
                "applied_to": "training fold only",
                "rows_before": int(len(y_train)),
                "rows_after": int(len(y_train_sm)),
            },
            "threshold": thr_info["threshold"],
            "threshold_objective": thr_info["objective"],
            "validation_metrics": thr_info["validation_metrics"],
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - t0, 2),
        }
        _log(
            f"  random_forest done in {time.time() - t0:.1f}s | "
            f"test recall={test_metrics['recall']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"pr_auc={test_metrics['pr_auc']:.4f}"
        )

    # ---------------- XGBoost -------------------------------------------- #
    if "xgboost" in models_to_train:
        _log("training xgboost (scale_pos_weight, early stopping on validation)")
        t0 = time.time()
        neg = int((y_train == 0).sum())
        pos = max(int((y_train == 1).sum()), 1)
        spw = neg / pos
        _log(f"  scale_pos_weight = {spw:.2f}")
        xgb = build_xgboost(spw)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        score_fn = SCORE_FUNCS["xgboost"]
        val_scores = score_fn(xgb, X_val)
        test_scores = score_fn(xgb, X_test)
        thr_info = select_threshold(y_val, val_scores)
        thr = thr_info["threshold"]
        test_metrics = evaluate_predictions(
            y_test, (test_scores >= thr).astype(int), test_scores
        )
        joblib.dump(xgb, artifacts_dir / "xgboost.joblib")
        metadata["models"]["xgboost"] = {
            **MODEL_INFO["xgboost"],
            "artifact": "xgboost.joblib",
            "trained": True,
            "hyperparameters": xgb.get_params(),
            "early_stopping": {
                "rounds": 40,
                "eval_metric": "aucpr",
                "best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
                "evaluated_on": "validation split",
            },
            "resampling": {
                "method": "scale_pos_weight",
                "value": round(spw, 4),
                "applied_to": "training fold only",
            },
            "threshold": thr_info["threshold"],
            "threshold_objective": thr_info["objective"],
            "validation_metrics": thr_info["validation_metrics"],
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - t0, 2),
        }
        _log(
            f"  xgboost done in {time.time() - t0:.1f}s | "
            f"test recall={test_metrics['recall']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"pr_auc={test_metrics['pr_auc']:.4f}"
        )

    # ---------------- Isolation Forest ----------------------------------- #
    if "isolation_forest" in models_to_train:
        _log("training isolation_forest (unsupervised, threshold from validation)")
        t0 = time.time()
        iso = build_isolation_forest()
        iso.fit(X_train)
        score_fn = SCORE_FUNCS["isolation_forest"]
        val_scores = score_fn(iso, X_val)
        test_scores = score_fn(iso, X_test)
        thr_info = select_threshold(y_val, val_scores)
        thr = thr_info["threshold"]
        test_metrics = evaluate_predictions(
            y_test, (test_scores >= thr).astype(int), test_scores
        )
        joblib.dump(iso, artifacts_dir / "isolation_forest.joblib")
        metadata["models"]["isolation_forest"] = {
            **MODEL_INFO["isolation_forest"],
            "artifact": "isolation_forest.joblib",
            "trained": True,
            "hyperparameters": iso.get_params(),
            "score_definition": (
                "-score_samples(X); higher = more anomalous. NOT a probability."
            ),
            "threshold": thr_info["threshold"],
            "threshold_objective": thr_info["objective"],
            "validation_metrics": thr_info["validation_metrics"],
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - t0, 2),
        }
        _log(
            f"  isolation_forest done in {time.time() - t0:.1f}s | "
            f"test recall={test_metrics['recall']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"pr_auc={test_metrics['pr_auc']:.4f}"
        )

    # ---------------- Recommend a model (by validation F2) --------------- #
    def _val_f2(m: dict[str, Any]) -> float:
        p = m["validation_metrics"]["precision"]
        r = m["validation_metrics"]["recall"]
        beta_sq = 4.0
        denom = beta_sq * p + r
        return (1 + beta_sq) * p * r / denom if denom else 0.0

    trained_models = {
        k: v for k, v in metadata["models"].items() if v.get("trained")
    }
    if trained_models:
        best = max(trained_models.items(), key=lambda kv: _val_f2(kv[1]))
        metadata["recommended_model"] = best[0]
        metadata["recommended_reason"] = (
            f"Highest validation F2 ({_val_f2(best[1]):.4f}); recall is "
            "prioritised while precision and false positives are monitored."
        )
    else:
        metadata["recommended_model"] = None
        metadata["recommended_reason"] = "No models were trained."

    metadata["training_seconds_total"] = round(time.time() - started, 2)

    # Persist the held-out test set so evaluate_models.py can re-verify the
    # reported numbers without ever touching the training path.
    test_frame = split.X_test.copy()
    test_frame["Class"] = split.y_test.values
    test_frame.to_parquet(artifacts_dir / "heldout_test.parquet", index=False)

    write_json(artifacts_dir / "metadata.json", metadata)
    _log(
        f"done in {metadata['training_seconds_total']}s | "
        f"recommended model: {metadata['recommended_model']}"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FraudShield AI models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--quick", action="store_true",
        help="Train on a small stratified subsample (smoke test only).",
    )
    parser.add_argument(
        "--models", nargs="*", choices=MODEL_ORDER, default=None,
        help="Restrict training to a subset of models.",
    )
    args = parser.parse_args()

    try:
        train_all(args.data, args.artifacts, quick=args.quick, only=args.models)
    except DatasetError as exc:
        print(f"\n[dataset error] {exc}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
