
"""Asynchronous database connection and session management."""

import ssl
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


def _prepare_database_url(raw_url: str) -> tuple[URL, ssl.SSLContext]:
    """Prepare a Neon PostgreSQL URL for asyncpg."""

    try:
        url = make_url(raw_url)
    except Exception:
        raise ValueError(
            "NEON_DATABASE_URL is not a valid database URL."
        ) from None

    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+asyncpg")
    elif url.drivername != "postgresql+asyncpg":
        raise ValueError("Only PostgreSQL URLs are supported.")

    query = dict(url.query)

    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)

    if str(sslmode or "require").lower() not in {
        "require", "verify-ca", "verify-full"
    }:
        raise ValueError("A secure TLS connection is required.")

    ssl_context = ssl.create_default_context()

    url = url.set(query=query)

    return url, ssl_context


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create and cache the asynchronous database engine."""

    if settings.neon_database_url is None:
        raise RuntimeError(
            "NEON_DATABASE_URL is not configured."
        )

    raw_url = settings.neon_database_url.get_secret_value().strip()

    if not raw_url:
        raise RuntimeError("NEON_DATABASE_URL is empty.")

    database_url, ssl_context = _prepare_database_url(raw_url)

    return create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=30,
        pool_recycle=1800,
        connect_args={"ssl": ssl_context},
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a reusable asynchronous session factory."""

    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db() -> AsyncIterator[AsyncSession]:
    """Provide a database session for FastAPI."""

    session_factory = get_session_factory()

    async with session_factory() as session:
        yield session


async def ping_database() -> int:
    """Verify database connectivity using SELECT 1."""

    async with get_engine().connect() as connection:
        result = await connection.execute(text("SELECT 1"))
        return result.scalar_one()


async def close_database() -> None:
    """Close database connections during shutdown."""

    if get_engine.cache_info().currsize == 0:
        return

    await get_engine().dispose()

    get_session_factory.cache_clear()
    get_engine.cache_clear()
