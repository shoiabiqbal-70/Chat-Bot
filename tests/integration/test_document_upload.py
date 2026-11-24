
"""Live integration tests for document upload and persistence."""

import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.api.documents import get_document_storage
from app.db.session import (
    close_database,
    get_session_factory,
)
from app.ingestion.storage import LocalDocumentStorage
from app.main import app
from app.models import Document


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION_TESTS") != "1",
    reason="Live database integration tests are disabled.",
)


UPLOAD_URL = "/api/documents/upload"


@pytest.fixture
def storage(tmp_path):
    """Provide isolated filesystem storage."""
    return LocalDocumentStorage(tmp_path / "uploads")


@pytest.fixture
def test_content():
    """Generate unique document content for each test."""
    return (
        f"Production RAG live integration test {uuid4()}"
    ).encode("utf-8")


@pytest.fixture
def client(storage, test_content):
    """Use the real database with temporary file storage."""
    checksum = hashlib.sha256(test_content).hexdigest()

    previous_overrides = app.dependency_overrides.copy()

    app.dependency_overrides[
        get_document_storage
    ] = lambda: storage

    try:
        with TestClient(app) as test_client:
            yield test_client

    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(
            previous_overrides
        )

        async def cleanup():
            """Remove only the document created by this test."""
            try:
                factory = get_session_factory()

                async with factory() as session:
                    await session.execute(
                        delete(Document).where(
                            Document.sha256 == checksum
                        )
                    )

                    await session.commit()

                    result = await session.execute(
                        select(Document.id).where(
                            Document.sha256 == checksum
                        )
                    )

                    assert result.scalar_one_or_none() is None

            finally:
                await close_database()

        asyncio.run(cleanup())


def upload(client, content, filename="integration.txt"):
    """Submit a real multipart upload request."""
    return client.post(
        UPLOAD_URL,
        files={
            "file": (
                filename,
                content,
                "text/plain",
            )
        },
    )


def test_live_document_registration(
    client,
    storage,
    test_content,
):
    """Register a document using the real database."""
    response = upload(client, test_content)

    assert response.status_code == 201, response.text

    data = response.json()
    document_id = UUID(data["id"])

    assert data["filename"] == "integration.txt"
    assert data["status"] == "pending"
    assert data["mime_type"] == "text/plain"

    assert data["sha256"] == hashlib.sha256(
        test_content
    ).hexdigest()

    assert storage.exists(document_id, ".txt")

    assert storage.get_path(
        document_id,
        ".txt",
    ).read_bytes() == test_content


def test_live_database_metadata(
    client,
    storage,
    test_content,
):
    """Verify committed metadata using a separate session."""
    response = upload(client, test_content)

    assert response.status_code == 201, response.text

    data = response.json()
    document_id = UUID(data["id"])

    async def verify():
        """Read committed metadata from PostgreSQL."""
        factory = get_session_factory()

        async with factory() as session:
            document = await session.get(
                Document,
                document_id,
            )

            assert document is not None
            assert document.filename == data["filename"]
            assert document.sha256 == data["sha256"]
            assert document.mime_type == data["mime_type"]
            assert document.status == "pending"
            assert document.created_at is not None
            assert document.updated_at is not None

    # Run database verification on the application's
    # event loop rather than creating another event loop.
    assert client.portal is not None
    client.portal.call(verify)

    assert storage.exists(document_id, ".txt")


def test_live_duplicate_upload(
    client,
    storage,
    test_content,
):
    """Reject repeated content even with a different filename."""
    first = upload(
        client,
        test_content,
        filename="first.txt",
    )

    assert first.status_code == 201, first.text

    second = upload(
        client,
        test_content,
        filename="renamed.txt",
    )

    assert second.status_code == 409, second.text

    assert len(list(storage.root.iterdir())) == 1


def test_live_concurrent_duplicate_upload(
    client,
    storage,
    test_content,
):
    """Allow only one registration for concurrent duplicates."""

    def submit(filename):
        """Upload identical content from a worker thread."""
        return upload(
            client,
            test_content,
            filename=filename,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(submit, "concurrent-a.txt")
        second = executor.submit(submit, "concurrent-b.txt")

        responses = [
            first.result(timeout=120),
            second.result(timeout=120),
        ]

    codes = sorted(
        response.status_code
        for response in responses
    )

    assert codes == [201, 409], [
        response.text
        for response in responses
    ]

    assert len(list(storage.root.iterdir())) == 1

    async def verify():
        """Confirm that only one database record exists."""
        checksum = hashlib.sha256(
            test_content
        ).hexdigest()

        factory = get_session_factory()

        async with factory() as session:
            result = await session.execute(
                select(Document).where(
                    Document.sha256 == checksum
                )
            )

            documents = result.scalars().all()

            assert len(documents) == 1

            return documents[0].id

    assert client.portal is not None
    document_id = client.portal.call(verify)

    assert storage.exists(document_id, ".txt")


def test_live_invalid_upload(
    client,
    storage,
    test_content,
):
    """Ensure invalid uploads leave no database or file records."""
    response = client.post(
        UPLOAD_URL,
        files={
            "file": (
                "invalid.pdf",
                test_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
    assert list(storage.root.iterdir()) == []

    async def verify():
        """Confirm that rejected content was not registered."""
        checksum = hashlib.sha256(
            test_content
        ).hexdigest()

        factory = get_session_factory()

        async with factory() as session:
            result = await session.execute(
                select(Document.id).where(
                    Document.sha256 == checksum
                )
            )

            assert result.scalar_one_or_none() is None

    assert client.portal is not None
    client.portal.call(verify)
