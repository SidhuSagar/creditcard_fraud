"""Model registry: loads trained artifacts once and serves inference.

Rules enforced here:
  * models are loaded from disk, never retrained per request,
  * a model counts as "available" only if ``metadata.json`` says it was trained
    *and* its artifact file actually exists on disk,
  * missing artifacts produce an explicit "not trained" state, never a
    fabricated prediction.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from training.preprocess import FEATURE_ORDER

logger = logging.getLogger("fraudshield.registry")

# Score semantics per model family
PROBABILITY_MODELS = {"random_forest", "xgboost"}
ANOMALY_MODELS = {"isolation_forest"}

# Alert severity cut-offs for probability models
SEVERITY_HIGH_PROB = 0.90
SEVERITY_MEDIUM_PROB = 0.70
# Relative cut-offs for anomaly scores (multiples of the decision threshold)
SEVERITY_HIGH_RATIO = 1.30
SEVERITY_MEDIUM_RATIO = 1.15


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but no usable artifact exists."""


@dataclass
class LoadedModel:
    key: str
    display_name: str
    family: str
    score_type: str
    artifact: str
    threshold: float
    entry: dict[str, Any]
    estimator: Any = field(repr=False, default=None)

    def is_anomaly(self) -> bool:
        return self.score_type == "anomaly_score"


class ModelRegistry:
    """Thread-safe, lazily-loaded collection of trained models."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir or settings.artifacts_dir)
        self._lock = threading.RLock()
        self._loaded = False
        self._metadata: dict[str, Any] = {}
        self._models: dict[str, LoadedModel] = {}
        self._preprocessor: Any = None
        self._load_error: str | None = None
        self._mtime_seen: float = -1.0

    # ---------------------------------------------------------------- load --
    def _metadata_mtime(self) -> float:
        p = self.artifacts_dir / "metadata.json"
        return p.stat().st_mtime if p.exists() else 0.0

    def load(self, force: bool = False) -> None:
        with self._lock:
            if self._loaded and not force:
                return
            self._metadata = {}
            self._models = {}
            self._preprocessor = None
            self._load_error = None

            meta_path = self.artifacts_dir / "metadata.json"
            if not meta_path.exists():
                self._load_error = (
                    f"No trained model metadata found at {meta_path}. Run "
                    "'python training/train_models.py' from the backend directory."
                )
                self._loaded = True
                logger.warning(self._load_error)
                return

            try:
                self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._load_error = f"Could not parse metadata.json: {exc}"
                self._loaded = True
                return

            pre_path = self.artifacts_dir / "preprocessor.joblib"
            if pre_path.exists():
                try:
                    self._preprocessor = joblib.load(pre_path)
                except Exception as exc:
                    self._load_error = f"Could not load preprocessor: {exc}"

            missing: list[str] = []
            for key, entry in self._metadata.get("models", {}).items():
                if not entry.get("trained"):
                    continue
                artifact = entry.get("artifact")
                if not artifact:
                    continue
                path = self.artifacts_dir / artifact
                if not path.exists():
                    missing.append(artifact)
                    continue
                try:
                    estimator = joblib.load(path)
                except Exception as exc:
                    logger.error("Failed to load %s: %s", artifact, exc)
                    missing.append(artifact)
                    continue
                self._models[key] = LoadedModel(
                    key=key,
                    display_name=entry.get("display_name", key),
                    family=entry.get("family", "supervised"),
                    score_type=entry.get("score_type", "probability"),
                    artifact=artifact,
                    threshold=float(entry.get("threshold", 0.5)),
                    entry=entry,
                    estimator=estimator,
                )

            if missing and not self._models:
                self._load_error = (
                    "Model metadata exists but the artifact files are missing "
                    f"({', '.join(sorted(set(missing)))}). Retrain the models."
                )
            self._loaded = True
            logger.info("registry loaded %d model(s): %s", len(self._models), list(self._models))

    def refresh_if_stale(self) -> None:
        """Reload when metadata.json changed on disk (e.g. after retraining)."""
        with self._lock:
            current = self._metadata_mtime()
            if not self._loaded or current != self._mtime_seen:
                self.load(force=True)
                self._mtime_seen = current

    # ------------------------------------------------------------- queries --
    @property
    def metadata(self) -> dict[str, Any]:
        self.load()
        return self._metadata

    @property
    def models(self) -> dict[str, LoadedModel]:
        self.load()
        return self._models

    @property
    def preprocessor(self) -> Any:
        self.load()
        return self._preprocessor

    @property
    def models_trained(self) -> bool:
        self.load()
        return bool(self._models) and self._preprocessor is not None

    @property
    def load_error(self) -> str | None:
        self.load()
        return self._load_error

    @property
    def recommended_model(self) -> str | None:
        self.load()
        rec = self._metadata.get("recommended_model")
        if rec in self._models:
            return rec
        return next(iter(self._models), None)

    def available_models(self) -> list[str]:
        return list(self.models.keys())

    def catalogue(self) -> list[dict[str, Any]]:
        """Every model the project defines, plus whether it is usable now."""
        self.load()
        declared = self._metadata.get("models", {})
        out: list[dict[str, Any]] = []
        for key, entries in declared.items():
            loaded = self._models.get(key)
            out.append(
                {
                    "key": key,
                    "display_name": entries.get("display_name", key),
                    "family": entries.get("family", "supervised"),
                    "score_type": entries.get("score_type", "probability"),
                    "description": entries.get("description", ""),
                    "trained": loaded is not None,
                    "artifact": entries.get("artifact"),
                }
            )
        if not out:
            # Nothing trained yet: still advertise the intended catalogue.
            from training.train_models import MODEL_INFO, MODEL_ORDER

            for key in MODEL_ORDER:
                info = MODEL_INFO[key]
                out.append({**info, "key": key, "trained": False, "artifact": None})
        return out

    def get(self, key: str) -> LoadedModel:
        models = self.models
        if key not in models:
            if not self.models_trained:
                raise ModelNotAvailableError(
                    self._load_error
                    or "No trained models are available. Train the models first."
                )
            raise ModelNotAvailableError(
                f"Model '{key}' is not available. Available: {sorted(models)}"
            )
        return models[key]

    # ---------------------------------------------------------- inference --
    def _matrix_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self._preprocessor is None:
            raise ModelNotAvailableError(
                "Preprocessing pipeline is missing; retrain the models."
            )
        # The preprocessor selects columns by NAME (fitted on a DataFrame), so
        # inference must also hand it a DataFrame with the canonical columns and
        # dtypes. This is what guarantees training/inference feature parity.
        ordered = frame.loc[:, FEATURE_ORDER].astype("float64")
        return np.asarray(self._preprocessor.transform(ordered), dtype=np.float64)

    def _to_matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        ordered = {col: [float(r[col]) for r in rows] for col in FEATURE_ORDER}
        frame = pd.DataFrame(ordered, columns=FEATURE_ORDER)
        return self._matrix_from_frame(frame)

    def _score(self, model: LoadedModel, X: np.ndarray) -> list[dict[str, Any]]:
        if model.score_type == "anomaly_score":
            scores = -model.estimator.score_samples(X)
        else:
            scores = model.estimator.predict_proba(X)[:, 1]

        results: list[dict[str, Any]] = []
        for score in np.asarray(scores, dtype=float):
            flagged = bool(score >= model.threshold)
            results.append(
                {
                    "prediction": 1 if flagged else 0,
                    "label": "Fraud" if flagged else "Legitimate",
                    "score": round(float(score), 6),
                    "score_type": model.score_type,
                    "threshold": model.threshold,
                    "model": model.key,
                    "model_display_name": model.display_name,
                    "alert_worthy": flagged,
                }
            )
        return results

    def predict(self, key: str, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Score one or many transactions from raw feature dicts."""
        if not rows:
            return []
        model = self.get(key)
        return self._score(model, self._to_matrix(rows))

    def predict_dataframe(
        self, key: str, frame: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Score a whole DataFrame in one vectorised pass.

        Used by the CSV uploader so that a 285k-row file is scored without
        building hundreds of thousands of intermediate Python dicts.
        """
        if frame is None or len(frame) == 0:
            return []
        model = self.get(key)
        return self._score(model, self._matrix_from_frame(frame))

    def severity_for(self, model: LoadedModel, score: float) -> str:
        """Map a decision score to a review severity bucket."""
        if model.score_type == "anomaly_score":
            if model.threshold <= 0:
                return "medium"
            ratio = score / model.threshold
            if ratio >= SEVERITY_HIGH_RATIO:
                return "high"
            if ratio >= SEVERITY_MEDIUM_RATIO:
                return "medium"
            return "low"
        if score >= SEVERITY_HIGH_PROB:
            return "high"
        if score >= SEVERITY_MEDIUM_PROB:
            return "medium"
        return "low"


# Module-level singleton used by the API layer.
registry = ModelRegistry()


def reset_registry(artifacts_dir: Path | None = None) -> ModelRegistry:
    """Rebuild the singleton - primarily for tests."""
    global registry
    registry = ModelRegistry(artifacts_dir)
    return registry
