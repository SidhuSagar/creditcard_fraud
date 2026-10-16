"""POST /api/analyze - upload a transaction CSV and classify every row."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas.analyze import AnalyzeResponse
from app.schemas.common import ErrorResponse
from app.services import analyze_service
from app.services.analyze_service import UploadError
from app.services.model_registry import ModelNotAvailableError
from training.train_models import MODEL_ORDER

router = APIRouter(tags=["analysis"])
logger = logging.getLogger("fraudshield.analyze")

_CHUNK = 1024 * 1024


async def _read_upload(file: UploadFile) -> bytes:
    """Read the upload, enforcing the size limit without trusting Content-Length."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "File is larger than the "
                    f"{settings.max_upload_bytes // (1024 * 1024)} MB upload limit."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Classify a transaction CSV as fraud or legitimate",
    responses={
        422: {"model": ErrorResponse, "description": "Unusable CSV"},
        503: {"model": ErrorResponse, "description": "Models not trained"},
    },
)
async def analyze(
    file: UploadFile = File(..., description="creditcard.csv-style transaction file"),
    model: str = Form(default="xgboost", description="Which trained model to use"),
) -> AnalyzeResponse:
    if model not in MODEL_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Available: {', '.join(MODEL_ORDER)}.",
        )

    content = await _read_upload(file)
    filename = file.filename or "upload.csv"

    try:
        payload = analyze_service.analyze_csv(content, filename, model)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyzeResponse(**payload)
