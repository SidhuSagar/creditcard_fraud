"""FraudShield AI - independent re-evaluation of the saved models.

Loads the persisted models + the persisted held-out test set and recomputes
metrics, then compares the freshly computed numbers with the numbers recorded
in ``metadata.json`` during training. Any drift is reported, which makes it
hard for fabricated or stale metrics to go unnoticed.

Usage:
    python backend/training/evaluate_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.metrics import evaluate_predictions  # noqa: E402
from training.preprocess import FEATURE_ORDER, write_json  # noqa: E402
from training.train_models import SCORE_FUNCS, MODEL_ORDER  # noqa: E402

DEFAULT_ARTIFACTS = BACKEND_DIR / "artifacts"


def _log(msg: str) -> None:
    print(f"[evaluate] {msg}", flush=True)


def evaluate_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    meta_path = artifacts_dir / "metadata.json"
    test_path = artifacts_dir / "heldout_test.parquet"
    pre_path = artifacts_dir / "preprocessor.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata at {meta_path}. Train the models first "
            "(python backend/training/train_models.py)."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"No held-out test set at {test_path}. Retrain the models."
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    preprocessor = joblib.load(pre_path)
    test_df = pd.read_parquet(test_path)
    y_test = test_df["Class"].to_numpy().astype(int)
    X_test = np.asarray(
        preprocessor.transform(test_df[FEATURE_ORDER]), dtype=np.float64
    )
    _log(f"held-out test set: {len(y_test):,} rows, {int(y_test.sum())} fraud")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(y_test)),
        "models": {},
        "matches_training_metadata": {},
        "threshold_source": "metadata.json (selected on validation during training)",
    }

    for key in MODEL_ORDER:
        entry = metadata.get("models", {}).get(key)
        if not entry or not entry.get("trained"):
            continue
        artifact = artifacts_dir / entry["artifact"]
        if not artifact.exists():
            report["models"][key] = {"error": f"missing artifact {artifact.name}"}
            continue

        model = joblib.load(artifact)
        scores = SCORE_FUNCS[key](model, X_test)
        thr = float(entry["threshold"])
        metrics = evaluate_predictions(y_test, (scores >= thr).astype(int), scores)

        report["models"][key] = {
            "display_name": entry.get("display_name", key),
            "threshold": thr,
            "score_type": entry.get("score_type"),
            "test_metrics": metrics,
        }

        recorded = entry.get("test_metrics", {})
        same = all(
            abs(float(recorded.get(m, -1)) - float(metrics[m])) < 1e-9
            for m in ("precision", "recall", "f1", "pr_auc")
        )
        report["matches_training_metadata"][key] = bool(same)
        _log(
            f"{key}: recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"f1={metrics['f1']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"| matches metadata: {same}"
        )

    write_json(artifacts_dir / "evaluation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate saved models.")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    try:
        report = evaluate_artifacts(args.artifacts)
    except FileNotFoundError as exc:
        print(f"\n[error] {exc}\n", file=sys.stderr)
        return 2
    drift = [k for k, ok in report["matches_training_metadata"].items() if not ok]
    if drift:
        print(f"[warn] metrics drifted from training metadata for: {drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
