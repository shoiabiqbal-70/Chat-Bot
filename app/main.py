"""Main entry point for the Production RAG Service."""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Document-based Retrieval-Augmented Generation service.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_router, prefix="/api")


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "message": "Welcome to the Production RAG Service",
        "documentation": "/docs",
        "health": "/api/health",
    }
