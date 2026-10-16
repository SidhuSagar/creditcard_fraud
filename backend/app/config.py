"""Application configuration.

Every environment-specific value is read from an environment variable with a
safe local default, so nothing sensitive or machine-specific lives in source.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project root
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class Settings:
    """Resolved runtime settings."""

    def __init__(self) -> None:
        self.project_root: Path = PROJECT_ROOT
        self.backend_dir: Path = BACKEND_DIR

        # Filesystem. The dataset is only needed to *train*; the web app scores
        # whatever CSV the user uploads, so it works without this file present.
        self.data_path: Path = _env_path(
            "FRAUDSHIELD_DATA_PATH", PROJECT_ROOT / "data" / "creditcard.csv"
        )
        self.artifacts_dir: Path = _env_path(
            "FRAUDSHIELD_ARTIFACTS_DIR", BACKEND_DIR / "artifacts"
        )

        # CORS - the frontend dev server origins
        origins = os.getenv(
            "FRAUDSHIELD_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:4173,http://127.0.0.1:4173,"
            "http://localhost:3000,http://127.0.0.1:3000",
        )
        self.cors_origins: list[str] = [o.strip() for o in origins.split(",") if o.strip()]

        # CSV upload limits. The public ULB/Kaggle creditcard.csv is ~144 MB
        # with ~285,000 rows, so the defaults must comfortably exceed that.
        self.max_upload_bytes: int = _env_int(
            "FRAUDSHIELD_MAX_UPLOAD_BYTES", 512 * 1024 * 1024
        )
        self.max_batch_rows: int = _env_int("FRAUDSHIELD_MAX_BATCH_ROWS", 500_000)
        # How many individual classified rows are sent back to the browser.
        # The summary always reflects every processed row.
        self.max_response_rows: int = _env_int("FRAUDSHIELD_MAX_RESPONSE_ROWS", 2_000)

    @property
    def dataset_available(self) -> bool:
        return self.data_path.exists() and self.data_path.stat().st_size > 0

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / "metadata.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
