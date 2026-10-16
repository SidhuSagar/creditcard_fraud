"""FraudShield AI - exploratory data analysis.

Writes a machine-readable ``artifacts/eda.json`` summary used by the docs and
by the dashboard's "original dataset" panel. No fabricated values: everything
is computed from the dataset on disk.

Usage:
    python backend/training/eda.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.preprocess import (  # noqa: E402
    FEATURE_ORDER,
    DatasetError,
    load_dataset,
    write_json,
)

PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "creditcard.csv"
DEFAULT_ARTIFACTS = BACKEND_DIR / "artifacts"


def run(data_path: Path, artifacts_dir: Path) -> dict:
    df, report = load_dataset(data_path)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    eda = {
        "dataset": report,
        "per_feature": {},
        "class_means": {},
        "amount_by_hour": _amount_by_hour(df),
    }
    for col in FEATURE_ORDER:
        eda["per_feature"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "fraud_mean": float(fraud[col].mean()),
            "legit_mean": float(legit[col].mean()),
        }
    eda["class_means"]["fraud_amount_mean"] = float(fraud["Amount"].mean())
    eda["class_means"]["legit_amount_mean"] = float(legit["Amount"].mean())

    write_json(artifacts_dir / "eda.json", eda)
    print(f"[eda] wrote {artifacts_dir / 'eda.json'}")
    print(
        f"[eda] {report['rows']:,} transactions | "
        f"{report['fraud']} fraud ({report['fraud_rate'] * 100:.4f}%)"
    )
    return eda


def _amount_by_hour(df) -> list[dict]:
    """Aggregate activity into 24 hourly buckets (Time is seconds from t0)."""
    hour = (df["Time"] // 3600 % 24).astype(int)
    out = []
    for h in range(24):
        mask = hour == h
        sub = df[mask]
        out.append(
            {
                "hour": h,
                "transactions": int(mask.sum()),
                "fraud": int(sub["Class"].sum()),
                "avg_amount": round(float(sub["Amount"].mean()), 2) if len(sub) else 0.0,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FraudShield AI EDA.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    try:
        run(args.data, args.artifacts)
    except DatasetError as exc:
        print(f"\n[dataset error] {exc}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
