from __future__ import annotations

import logging

from fastapi import APIRouter

from app import __version__
from app.config import settings
from app.schemas.common import HealthResponse
from app.services.model_registry import registry
from app.services import generic_detector

router = APIRouter(tags=["health"])
logger = logging.getLogger("fraudshield.health")


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    registry.refresh_if_stale()
    return HealthResponse(
        status="ok" if registry.models_trained else "degraded",
        version=__version__,
        dataset_configured=settings.dataset_available,
        models_trained=registry.models_trained,
        available_models=registry.available_models(),
        recommended_model=registry.recommended_model,
        detail=registry.load_error,
        generic_mode_available=generic_detector.available(),
    )
