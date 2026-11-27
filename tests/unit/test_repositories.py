"""Unit tests for document and chunk repositories."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models import Document, Chunk
from app.repositories import DocumentRepository, ChunkRepository


def make_session():
    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


def make_result(items=None, scalar=None):
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalar_one.return_value = scalar
    result.scalars.return_value.all.return_value = items or []
    return result


# DOCUMENT REPOSITORY TESTS

def test_document_create():
    async def run():
        session = make_session()
        document = MagicMock(spec=Document)

        result = await DocumentRepository(session).create(document)

        assert result is document
        session.add.assert_called_once_with(document)
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once_with(document)

    asyncio.run(run())


def test_document_get_by_id():
    async def run():
        session = make_session()
        document = MagicMock(spec=Document)
        session.execute.return_value = make_result(scalar=document)

        result = await DocumentRepository(session).get_by_id(uuid4())

        assert result is document

    asyncio.run(run())


def test_document_get_by_id_not_found():
    async def run():
        session = make_session()
        session.execute.return_value = make_result()

        result = await DocumentRepository(session).get_by_id(uuid4())

        assert result is None

    asyncio.run(run())


def test_document_get_by_sha256():
    async def run():
        session = make_session()
        document = MagicMock(spec=Document)
        session.execute.return_value = make_result(scalar=document)

        result = await DocumentRepository(session).get_by_sha256(
            "a" * 64
        )

        assert result is document

    asyncio.run(run())


def test_document_list_and_pagination():
    async def run():
        session = make_session()
        documents = [MagicMock(spec=Document)]

        session.execute.return_value = make_result(items=documents)

        result = await DocumentRepository(session).list_documents(
            limit=10,
            offset=5,
        )

        assert result == documents

    asyncio.run(run())


def test_document_list_with_status_filter():
    async def run():
        session = make_session()
        session.execute.return_value = make_result()

        result = await DocumentRepository(session).list_documents(
            status="completed"
        )

        assert result == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "limit, offset",
    [(0, 0), (501, 0), (10, -1)],
)
def test_document_invalid_pagination(limit, offset):
    async def run():
        session = make_session()

        with pytest.raises(ValueError):
            await DocumentRepository(session).list_documents(
                limit=limit,
                offset=offset,
            )

    asyncio.run(run())


def test_document_update_status():
    async def run():
        session = make_session()
        document = MagicMock(spec=Document)
        session.execute.return_value = make_result(scalar=document)

        result = await DocumentRepository(session).update_status(
            document_id=uuid4(),
            status="completed",
        )

        assert result is document
        assert document.status == "completed"
        assert document.error_message is None
        session.flush.assert_awaited_once()

    asyncio.run(run())


def test_document_update_missing_document():
    async def run():
        session = make_session()
        session.execute.return_value = make_result()

        result = await DocumentRepository(session).update_status(
            uuid4(),
            "completed",
        )

        assert result is None
        session.flush.assert_not_awaited()

    asyncio.run(run())


def test_document_delete():
    async def run():
        session = make_session()
        document = MagicMock(spec=Document)
        session.execute.return_value = make_result(scalar=document)

        result = await DocumentRepository(session).delete(uuid4())

        assert result is True
        session.delete.assert_awaited_once_with(document)
        session.flush.assert_awaited_once()

    asyncio.run(run())


def test_document_delete_missing():
    async def run():
        session = make_session()
        session.execute.return_value = make_result()

        result = await DocumentRepository(session).delete(uuid4())

        assert result is False
        session.delete.assert_not_awaited()

    asyncio.run(run())


# CHUNK REPOSITORY TESTS

def test_chunk_create_many():
    async def run():
        session = make_session()
        chunks = [MagicMock(spec=Chunk), MagicMock(spec=Chunk)]

        result = await ChunkRepository(session).create_many(chunks)

        assert result == chunks
        session.add_all.assert_called_once_with(chunks)
        session.flush.assert_awaited_once()

    asyncio.run(run())


def test_chunk_create_many_empty():
    async def run():
        session = make_session()

        result = await ChunkRepository(session).create_many([])

        assert result == []
        session.add_all.assert_not_called()
        session.flush.assert_not_awaited()

    asyncio.run(run())


def test_chunk_get_by_id():
    async def run():
        session = make_session()
        chunk = MagicMock(spec=Chunk)
        session.execute.return_value = make_result(scalar=chunk)

        result = await ChunkRepository(session).get_by_id(uuid4())

        assert result is chunk

    asyncio.run(run())


def test_chunk_list_for_document():
    async def run():
        session = make_session()
        chunks = [
            MagicMock(spec=Chunk),
            MagicMock(spec=Chunk),
        ]
        session.execute.return_value = make_result(items=chunks)

        result = await ChunkRepository(session).list_for_document(
            uuid4()
        )

        assert result == chunks

    asyncio.run(run())


@pytest.mark.parametrize(
    "limit, offset",
    [(0, 0), (1001, 0), (10, -1)],
)
def test_chunk_invalid_pagination(limit, offset):
    async def run():
        session = make_session()

        with pytest.raises(ValueError):
            await ChunkRepository(session).list_for_document(
                uuid4(),
                limit=limit,
                offset=offset,
            )

    asyncio.run(run())


def test_chunk_count():
    async def run():
        session = make_session()
        result = make_result(scalar=7)
        session.execute.return_value = result

        count = await ChunkRepository(session).count_for_document(
            uuid4()
        )

        assert count == 7

    asyncio.run(run())


def test_chunk_delete_for_document():
    async def run():
        session = make_session()
        result = MagicMock()
        result.rowcount = 3
        session.execute.return_value = result

        deleted = await ChunkRepository(session).delete_for_document(
            uuid4()
        )

        assert deleted == 3
        session.flush.assert_awaited_once()

    asyncio.run(run())
