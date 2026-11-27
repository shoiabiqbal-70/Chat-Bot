"""Opt-in integration tests for the Neon PostgreSQL database."""

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import delete, insert, select, text
from sqlalchemy.exc import IntegrityError

from app.db.session import get_engine, close_database
from app.models import Document, Chunk


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION_TESTS") != "1",
    reason="Live database integration tests are disabled.",
)


@asynccontextmanager
async def isolated_connection():
    """Provide a database connection with guaranteed rollback."""

    try:
        async with get_engine().connect() as connection:
            transaction = await connection.begin()

            try:
                yield connection
            finally:
                if transaction.is_active:
                    await transaction.rollback()
    finally:
        await close_database()


async def create_document(connection, **overrides):
    """Insert an isolated test document and return its UUID."""

    document_id = uuid4()

    values = {
        "id": document_id,
        "filename": "integration-test.pdf",
        "sha256": hashlib.sha256(
            document_id.bytes
        ).hexdigest(),
        "mime_type": "application/pdf",
    }

    values.update(overrides)

    await connection.execute(
        insert(Document.__table__).values(**values)
    )

    return values["id"]


async def create_chunk(connection, document_id, **overrides):
    """Insert an isolated test chunk and return its UUID."""

    chunk_id = uuid4()

    values = {
        "id": chunk_id,
        "document_id": document_id,
        "text": "Database integration test content.",
        "page_number": 1,
        "chunk_index": 0,
        "token_count": 5,
    }

    values.update(overrides)

    await connection.execute(
        insert(Chunk.__table__).values(**values)
    )

    return values["id"]


async def expect_integrity_error(
    connection, statement, expected_sqlstate
):
    """Verify a database constraint using an isolated savepoint."""

    with pytest.raises(IntegrityError) as captured:
        async with connection.begin_nested():
            await connection.execute(statement)

    assert captured.value.orig.sqlstate == expected_sqlstate


def test_database_connectivity():
    """DB-001: Verify live PostgreSQL connectivity."""

    async def run():
        async with isolated_connection() as connection:
            assert await connection.scalar(text("SELECT 1")) == 1

    asyncio.run(run())


def test_document_insertion():
    """DB-002: Verify document insertion and retrieval."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            result = await connection.scalar(
                select(Document.__table__.c.filename)
                .where(Document.__table__.c.id == document_id)
            )

            assert result == "integration-test.pdf"

    asyncio.run(run())


def test_default_document_status():
    """DB-003: Verify the database default document status."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            status = await connection.scalar(
                select(Document.__table__.c.status)
                .where(Document.__table__.c.id == document_id)
            )

            assert status == "pending"

    asyncio.run(run())


def test_duplicate_sha256_rejected():
    """DB-004: Verify unique document hash enforcement."""

    async def run():
        async with isolated_connection() as connection:
            digest = hashlib.sha256(uuid4().bytes).hexdigest()

            await create_document(connection, sha256=digest)

            statement = insert(Document.__table__).values(
                id=uuid4(),
                filename="duplicate.pdf",
                sha256=digest,
                mime_type="application/pdf",
            )

            await expect_integrity_error(
                connection, statement, "23505"
            )

    asyncio.run(run())


def test_invalid_document_status_rejected():
    """DB-005: Verify document status constraint enforcement."""

    async def run():
        async with isolated_connection() as connection:
            statement = insert(Document.__table__).values(
                id=uuid4(),
                filename="invalid-status.pdf",
                sha256=hashlib.sha256(uuid4().bytes).hexdigest(),
                mime_type="application/pdf",
                status="invalid",
            )

            await expect_integrity_error(
                connection, statement, "23514"
            )

    asyncio.run(run())


def test_chunk_insertion():
    """DB-006: Verify chunk insertion and retrieval."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)
            chunk_id = await create_chunk(connection, document_id)

            result = await connection.scalar(
                select(Chunk.__table__.c.text)
                .where(Chunk.__table__.c.id == chunk_id)
            )

            assert result == "Database integration test content."

    asyncio.run(run())


def test_foreign_key_enforcement():
    """DB-007: Reject chunks without a valid parent document."""

    async def run():
        async with isolated_connection() as connection:
            statement = insert(Chunk.__table__).values(
                id=uuid4(),
                document_id=uuid4(),
                text="Orphan chunk",
                chunk_index=0,
            )

            await expect_integrity_error(
                connection, statement, "23503"
            )

    asyncio.run(run())


def test_duplicate_chunk_index_rejected():
    """DB-008: Verify unique chunk indexes per document."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            await create_chunk(connection, document_id)

            statement = insert(Chunk.__table__).values(
                id=uuid4(),
                document_id=document_id,
                text="Duplicate chunk",
                chunk_index=0,
            )

            await expect_integrity_error(
                connection, statement, "23505"
            )

    asyncio.run(run())


def test_negative_chunk_index_rejected():
    """DB-009: Reject negative chunk indexes."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            statement = insert(Chunk.__table__).values(
                id=uuid4(),
                document_id=document_id,
                text="Invalid chunk",
                chunk_index=-1,
            )

            await expect_integrity_error(
                connection, statement, "23514"
            )

    asyncio.run(run())


def test_invalid_page_number_rejected():
    """DB-010: Reject invalid page numbers."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            statement = insert(Chunk.__table__).values(
                id=uuid4(),
                document_id=document_id,
                text="Invalid page",
                page_number=0,
                chunk_index=0,
            )

            await expect_integrity_error(
                connection, statement, "23514"
            )

    asyncio.run(run())


def test_cascade_deletion():
    """DB-011: Verify database-level cascading deletion."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            chunk_id = await create_chunk(
                connection, document_id
            )

            await connection.execute(
                delete(Document.__table__)
                .where(Document.__table__.c.id == document_id)
            )

            result = await connection.scalar(
                select(Chunk.__table__.c.id)
                .where(Chunk.__table__.c.id == chunk_id)
            )

            assert result is None

    asyncio.run(run())


def test_transaction_rollback():
    """DB-012: Verify rolled-back records are not persisted."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)

            result = await connection.scalar(
                select(Document.__table__.c.id)
                .where(Document.__table__.c.id == document_id)
            )

            assert result == document_id

        async with isolated_connection() as connection:
            result = await connection.scalar(
                select(Document.__table__.c.id)
                .where(Document.__table__.c.id == document_id)
            )

            assert result is None

    asyncio.run(run())


def test_timestamp_defaults():
    """DB-013: Verify database timestamp defaults."""

    async def run():
        async with isolated_connection() as connection:
            document_id = await create_document(connection)
            chunk_id = await create_chunk(connection, document_id)

            result = await connection.execute(
                select(
                    Document.__table__.c.created_at,
                    Document.__table__.c.updated_at,
                ).where(
                    Document.__table__.c.id == document_id
                )
            )

            created_at, updated_at = result.one()

            chunk_created_at = await connection.scalar(
                select(Chunk.__table__.c.created_at)
                .where(Chunk.__table__.c.id == chunk_id)
            )

            assert created_at.utcoffset() is not None
            assert updated_at.utcoffset() is not None
            assert chunk_created_at.utcoffset() is not None

    asyncio.run(run())
