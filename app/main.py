
"""Main entry point for the Production RAG Service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import settings
from app.db.session import close_database


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    try:
        yield
    finally:
        await close_database()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Document-based Retrieval-Augmented "
        "Generation service."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.include_router(
    health_router,
    prefix="/api",
)

app.include_router(
    documents_router,
    prefix="/api",
)


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "message": "Welcome to the Production RAG Service",
        "documentation": "/docs",
        "health": "/api/health",
    }
