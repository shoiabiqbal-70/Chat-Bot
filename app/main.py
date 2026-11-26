
"""Main entry point for the Production RAG Service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.documents import router as documents_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.questions import router as questions_router
from app.config import settings
from app.db.session import close_database


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    if settings.app_env == "production":
        required = {
            "NEON_DATABASE_URL": settings.neon_database_url,
            "NEON_MIGRATION_DATABASE_URL": settings.neon_migration_database_url,
            "PINECONE_API_KEY": settings.pinecone_api_key,
            "GROQ_API_KEY": settings.groq_api_key,
        }
        missing = [name for name, value in required.items()
                   if value is None or not value.get_secret_value().strip()]
        if missing:
            raise RuntimeError("Missing production settings: " + ", ".join(missing))
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

logger = logging.getLogger(__name__)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    """Log request identity, status, and latency without payloads or secrets."""
    request_id = uuid4().hex
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed id=%s method=%s path=%s duration_ms=%.1f",
                         request_id, request.method, request.url.path,
                         (perf_counter() - started) * 1000)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info("request id=%s method=%s path=%s status=%s duration_ms=%.1f",
                request_id, request.method, request.url.path,
                response.status_code, (perf_counter() - started) * 1000)
    return response


app.include_router(
    health_router,
    prefix="/api",
)
app.include_router(auth_router, prefix="/api")

app.include_router(
    documents_router,
    prefix="/api",
)
app.include_router(questions_router, prefix="/api")

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/app", tags=["General"])
async def workspace() -> FileResponse:
    """Serve the document workspace."""
    return FileResponse(FRONTEND / "index.html")


@app.get("/", tags=["General"], response_model=None)
async def root() -> dict[str, str] | RedirectResponse:
    """Open the hosted workspace and preserve local API discovery."""
    if settings.app_env == "production":
        return RedirectResponse(url="/app")
    return {"message": "Welcome to the Production RAG Service",
            "documentation": "/docs", "health": "/api/health",
            "workspace": "/app"}
