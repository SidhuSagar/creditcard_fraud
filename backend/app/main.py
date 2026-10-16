"""FraudShield AI - FastAPI application entry point.

A single-purpose service: upload a CSV of transactions and get every row
classified as *fraud* or *legitimate* by the trained model.

Run from the ``backend`` directory:
    uvicorn app.main:app --reload --port 8010

Interactive API docs: http://localhost:8010/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.routes import api_router
from app.services.model_registry import registry
from training.preprocess import DatasetError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("fraudshield")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("starting FraudShield AI v%s", __version__)
    registry.load()
    logger.info("dataset configured : %s", settings.dataset_available)
    logger.info("models trained     : %s", registry.models_trained)
    if registry.load_error:
        logger.warning("registry: %s", registry.load_error)
    yield
    logger.info("shutting down FraudShield AI")


app = FastAPI(
    title="FraudShield AI API",
    description=(
        "Credit-card fraud detection API. Upload a transaction CSV and every "
        "row is classified as fraud or legitimate. Educational project - it "
        "does not connect to banking systems and does not block real "
        "transactions.\n\n"
        "Predictions come from models trained on the public ULB/Kaggle Credit "
        "Card Fraud dataset. See /docs for the full schema."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["meta"], summary="Service banner")
def root() -> dict:
    return {
        "name": "FraudShield AI",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
        "analyze": "/api/analyze",
        "disclaimer": (
            "Educational machine-learning project. Not connected to any real "
            "banking or payment system."
        ),
    }


# --------------------------------------------------------------------------- #
# Error handling: always return a readable, structured message
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_handler(_req: Request, exc: RequestValidationError) -> JSONResponse:
    problems = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []) if p != "body")
        problems.append(f"{loc or 'body'}: {err.get('msg')}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed: " + "; ".join(problems),
            "code": "validation_error",
            "context": {"errors": exc.errors()[:20]},
        },
    )


@app.exception_handler(DatasetError)
async def dataset_handler(_req: Request, exc: DatasetError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "code": "dataset_error"},
    )


@app.exception_handler(Exception)
async def unhandled_handler(_req: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred. Check the backend logs.",
            "code": "internal_error",
        },
    )
