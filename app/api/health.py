"""Health-check API endpoints."""

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check application health",
)
async def health_check() -> HealthResponse:
    """Return basic application health information.

    Returns:
        HealthResponse: Application status and configuration details.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        environment=settings.app_env,
    )
