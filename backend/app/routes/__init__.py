"""API routers."""
from fastapi import APIRouter

from app.routes import analyze, health

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(analyze.router)

__all__ = ["api_router"]
