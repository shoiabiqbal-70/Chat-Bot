
"""Unit tests for document registration service."""

import asyncio
import hashlib
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.datastructures import Headers

from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.ingestion.validation import UploadValidationError
from app.models import Document
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
    DuplicateDocumentError,
    is_duplicate_hash_error,
)


CONTENT = b"Production RAG document registration test."


@pytest.fixture
def session():
    """Provide a mocked asynchronous database session."""
    session = MagicMock()

    session.in_transaction.return_value = False
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none.return_value = None

    session.execute.return_value = result

    return session


@pytest.fixture
def storage(tmp_path):
    """Provide isolated local document storage."""
    return LocalDocumentStorage(tmp_path / "uploads")


def register(
    service,
    content=CONTENT,
    filename="sample.txt",
    mime_type="text/plain",
):
    """Run registration and close the uploaded file."""

    async def run():
        upload = UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({
                "content-type": mime_type,
            }),
        )

        try:
            return await service.register(upload)
        finally:
            await upload.close()

    return asyncio.run(run())


def test_successful_registration(session, storage):
    """Register metadata and persist the original file."""
    service = DocumentService(session, storage)

    document = register(service)

    assert isinstance(document, Document)
    assert document.id is not None
    assert document.filename == "sample.txt"
    assert document.status == "pending"

    assert document.sha256 == hashlib.sha256(
        CONTENT
    ).hexdigest()

    assert storage.exists(document.id, ".txt")

    assert storage.get_path(
        document.id,
        ".txt",
    ).read_bytes() == CONTENT

    session.add.assert_called_once_with(document)
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


def test_duplicate_document(session, storage):
    """Reject documents with an existing content hash."""
    session.execute.return_value.scalar_one_or_none.return_value = (
        MagicMock(spec=Document)
    )

    service = DocumentService(session, storage)

    with pytest.raises(DuplicateDocumentError):
        register(service)

    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()

    assert list(storage.root.iterdir()) == []


def test_invalid_upload(session, storage):
    """Reject invalid files before database operations."""
    service = DocumentService(session, storage)

    with pytest.raises(UploadValidationError):
        register(
            service,
            content=b"Invalid PDF",
            filename="invalid.pdf",
            mime_type="application/pdf",
        )

    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []


def test_database_query_failure(session, storage):
    """Rollback when duplicate detection fails."""
    session.execute.side_effect = SQLAlchemyError(
        "Simulated database failure"
    )

    service = DocumentService(session, storage)

    with pytest.raises(DocumentServiceError):
        register(service)

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []


def test_database_insert_failure(session, storage):
    """Avoid storing a file when metadata insertion fails."""
    session.flush.side_effect = SQLAlchemyError(
        "Simulated insert failure"
    )

    service = DocumentService(session, storage)

    with pytest.raises(DocumentServiceError):
        register(service)

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []


def test_storage_failure(session, storage, monkeypatch):
    """Rollback metadata when file storage fails."""

    def fail_save(*args):
        """Simulate a storage failure."""
        raise StorageError("Simulated storage failure")

    monkeypatch.setattr(storage, "save", fail_save)

    service = DocumentService(session, storage)

    with pytest.raises(DocumentServiceError):
        register(service)

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []


def test_commit_failure(session, storage):
    """Preserve the file when commit outcome is uncertain."""
    session.commit.side_effect = SQLAlchemyError(
        "Simulated commit failure"
    )

    service = DocumentService(session, storage)

    with pytest.raises(DocumentServiceError):
        register(service)

    session.rollback.assert_awaited_once()

    files = list(storage.root.iterdir())

    assert len(files) == 1
    assert files[0].read_bytes() == CONTENT


def test_existing_transaction_rejected(session, storage):
    """Prevent accidental commits of unrelated transactions."""
    session.in_transaction.return_value = True

    service = DocumentService(session, storage)

    with pytest.raises(DocumentServiceError):
        register(service)

    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []


class SimulatedConstraintError(Exception):
    """Represent a PostgreSQL constraint violation."""

    constraint_name = "uq_documents_sha256"


def test_duplicate_constraint_detection():
    """Recognize the document hash uniqueness constraint."""
    error = IntegrityError(
        "INSERT",
        {},
        SimulatedConstraintError(),
    )

    assert is_duplicate_hash_error(error)


def test_unrelated_constraint_detection():
    """Do not misclassify unrelated integrity failures."""
    error = IntegrityError(
        "INSERT",
        {},
        Exception("Other constraint"),
    )

    assert not is_duplicate_hash_error(error)


def test_concurrent_duplicate_rejection(session, storage):
    """Handle a uniqueness violation during insertion."""
    session.flush.side_effect = IntegrityError(
        "INSERT",
        {},
        SimulatedConstraintError(),
    )

    service = DocumentService(session, storage)

    with pytest.raises(DuplicateDocumentError):
        register(service)

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()

    assert list(storage.root.iterdir()) == []
