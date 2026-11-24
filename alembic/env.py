"""Configure asynchronous Alembic migrations for Neon PostgreSQL."""

import asyncio
import ssl
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection, URL, make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.db.base import Base
import app.models


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_migration_url() -> URL:
    """Return a validated direct PostgreSQL migration URL."""

    secret = settings.neon_migration_database_url

    if secret is None:
        raise RuntimeError(
            "NEON_MIGRATION_DATABASE_URL is not configured."
        )

    raw_url = secret.get_secret_value().strip()

    if not raw_url:
        raise RuntimeError("Migration database URL is empty.")

    try:
        url = make_url(raw_url)
    except Exception:
        raise ValueError(
            "Invalid migration database URL."
        ) from None

    if url.drivername not in {
        "postgres",
        "postgresql",
        "postgresql+asyncpg",
    }:
        raise ValueError("Only PostgreSQL URLs are supported.")

    if not url.host or "-pooler" in url.host.lower():
        raise ValueError(
            "A direct Neon database connection is required."
        )

    query = dict(url.query)
    query.pop("sslmode", None)
    query.pop("channel_binding", None)

    return url.set(
        drivername="postgresql+asyncpg",
        query=query,
    )


def run_migrations_offline() -> None:
    """Generate PostgreSQL SQL without connecting to Neon."""

    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations through an established connection."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create and dispose a temporary migration engine."""

    ssl_context = ssl.create_default_context()

    engine = create_async_engine(
        get_migration_url(),
        poolclass=pool.NullPool,
        connect_args={"ssl": ssl_context},
        echo=False,
    )

    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    """Execute asynchronous migrations."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
