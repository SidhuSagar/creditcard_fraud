from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    dataset_configured: bool
    models_trained: bool
    available_models: list[str]
    recommended_model: str | None = None
    detail: str | None = None
    generic_mode_available: bool = False


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    context: dict[str, Any] | None = Field(default=None)
