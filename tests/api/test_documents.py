
"""HTTP API tests for document upload and registration."""

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.documents import get_document_storage
from app.db.session import get_db
from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.main import app
from app.models import Document


def test_atomic_ingest_returns_completed_document(client, session, monkeypatch):
    """The main upload route must report success only after indexing."""
    # Stub indexing so this route test checks HTTP semantics independently of
    # the embedding and vector providers.
    from app.api import documents as module
    from app.repositories import DocumentRepository

    async def fake_process(document_id, db, storage, vectors):
        document = session.add.call_args.args[0]
        document.status = "completed"
        return 3

    async def fake_get_by_id(self, document_id):
        return session.add.call_args.args[0]

    monkeypatch.setattr(module, "process_document", fake_process)
    monkeypatch.setattr(module, "VectorStore", lambda: object())
    monkeypatch.setattr(DocumentRepository, "get_by_id", fake_get_by_id)
    response = client.post("/api/documents", files={
        "file": ("sample.txt", CONTENT, "text/plain")
    })
    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert response.json()["chunk_count"] == 3


UPLOAD_URL = "/api/documents/upload"
DOCUMENTS_URL = "/api/documents"
CONTENT = b"Production RAG API integration test."


def test_list_documents_empty(client, session):
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute.return_value = result
    response = client.get(DOCUMENTS_URL)
    assert response.status_code == 200
    assert response.json() == []


def test_get_missing_document(client, session):
    session.execute.return_value.scalar_one_or_none.return_value = None
    response = client.get(f"{DOCUMENTS_URL}/{UUID(int=1)}")
    assert response.status_code == 404


def test_list_documents_rejects_invalid_pagination(client):
    response = client.get(f"{DOCUMENTS_URL}?limit=0")
    assert response.status_code == 422

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document"
)


@pytest.fixture
def session():
    """Provide a mocked asynchronous database session."""
    session = MagicMock()

    session.in_transaction.return_value = False
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    async def refresh(document):
        """Simulate database-generated timestamps."""
        now = datetime.now(timezone.utc)
        document.created_at = now
        document.updated_at = now

    session.refresh = AsyncMock(side_effect=refresh)

    return session


@pytest.fixture
def storage(tmp_path):
    """Provide isolated temporary document storage."""
    return LocalDocumentStorage(tmp_path / "uploads")


@pytest.fixture
def client(session, storage):
    """Provide an API client with isolated dependencies."""

    async def override_database():
        """Yield the mocked database session."""
        yield session

    def override_storage():
        """Return temporary document storage."""
        return storage

    previous_overrides = app.dependency_overrides.copy()

    app.dependency_overrides[get_db] = override_database

    app.dependency_overrides[
        get_document_storage
    ] = override_storage

    try:
        with TestClient(app) as test_client:
            yield test_client

    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(
            previous_overrides
        )


def upload(
    client,
    filename="sample.txt",
    content=CONTENT,
    mime_type="text/plain",
):
    """Submit a multipart document upload request."""
    return client.post(
        UPLOAD_URL,
        files={
            "file": (
                filename,
                content,
                mime_type,
            )
        },
    )


def make_docx_fixture():
    """Create a minimal DOCX structural test fixture."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "<document/>",
        )

    return buffer.getvalue()


def test_successful_text_upload(
    client,
    session,
    storage,
):
    """Return HTTP 201 and persist the uploaded document."""
    response = upload(client)

    assert response.status_code == 201

    data = response.json()
    document_id = UUID(data["id"])

    assert data["filename"] == "sample.txt"
    assert data["mime_type"] == "text/plain"
    assert data["status"] == "pending"

    assert data["sha256"] == hashlib.sha256(
        CONTENT
    ).hexdigest()

    assert datetime.fromisoformat(
        data["created_at"].replace("Z", "+00:00")
    ).tzinfo is not None

    assert storage.exists(document_id, ".txt")

    assert storage.get_path(
        document_id,
        ".txt",
    ).read_bytes() == CONTENT

    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


def test_pdf_upload(client, storage):
    """Accept a document with the expected PDF signature."""
    content = b"%PDF-1.7\nTest PDF signature\n%%EOF"

    response = upload(
        client,
        filename="sample.pdf",
        content=content,
        mime_type="application/pdf",
    )

    assert response.status_code == 201

    document_id = UUID(response.json()["id"])

    assert storage.exists(document_id, ".pdf")


def test_docx_upload(client, storage):
    """Accept a DOCX structural validation fixture."""
    response = upload(
        client,
        filename="sample.docx",
        content=make_docx_fixture(),
        mime_type=DOCX_MIME,
    )

    assert response.status_code == 201

    document_id = UUID(response.json()["id"])

    assert storage.exists(document_id, ".docx")


def test_duplicate_upload(
    client,
    session,
    storage,
):
    """Return HTTP 409 for duplicate document content."""
    session.execute.return_value.scalar_one_or_none.return_value = (
        MagicMock(spec=Document)
    )

    response = upload(client)

    assert response.status_code == 409
    assert "identical content" in response.json()["detail"]

    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()

    assert list(storage.root.iterdir()) == []


def test_empty_upload(client, session):
    """Return HTTP 400 for empty documents."""
    response = upload(
        client,
        content=b"",
    )

    assert response.status_code == 400

    session.execute.assert_not_awaited()


def test_unsupported_extension(client, session):
    """Return HTTP 415 for unsupported file extensions."""
    response = upload(
        client,
        filename="sample.exe",
        mime_type="application/octet-stream",
    )

    assert response.status_code == 415

    session.execute.assert_not_awaited()


def test_invalid_pdf(client, session):
    """Return HTTP 415 for an invalid PDF signature."""
    response = upload(
        client,
        filename="invalid.pdf",
        content=b"Not a PDF",
        mime_type="application/pdf",
    )

    assert response.status_code == 415

    session.execute.assert_not_awaited()


def test_mime_type_mismatch(client, session):
    """Reject mismatched file extensions and MIME types."""
    response = upload(
        client,
        filename="sample.txt",
        mime_type="application/pdf",
    )

    assert response.status_code == 415

    session.execute.assert_not_awaited()


def test_missing_upload_field(client):
    """Return HTTP 422 when the file field is missing."""
    response = client.post(UPLOAD_URL)

    assert response.status_code == 422


def test_oversized_upload(
    client,
    session,
    monkeypatch,
):
    """Return HTTP 413 when the upload exceeds the limit."""
    monkeypatch.setattr(
        "app.ingestion.validation.settings.max_upload_size_mb",
        1,
    )

    response = upload(
        client,
        content=b"A" * (1024 * 1024 + 1),
    )

    assert response.status_code == 413

    session.execute.assert_not_awaited()


def test_database_failure(
    client,
    session,
    storage,
):
    """Return HTTP 503 without exposing database errors."""
    session.execute.side_effect = SQLAlchemyError(
        "Private database failure details"
    )

    response = upload(client)

    assert response.status_code == 503

    assert response.json()["detail"] == (
        "Document registration is unavailable."
    )

    assert "Private database" not in response.text

    session.rollback.assert_awaited_once()

    assert list(storage.root.iterdir()) == []


def test_storage_failure(
    client,
    session,
    storage,
    monkeypatch,
):
    """Return HTTP 503 when document storage fails."""

    def fail_save(*args):
        """Simulate a filesystem failure."""
        raise StorageError("Private storage failure")

    monkeypatch.setattr(
        storage,
        "save",
        fail_save,
    )

    response = upload(client)

    assert response.status_code == 503
    assert "Private storage failure" not in response.text

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


def test_response_excludes_internal_fields(client):
    """Prevent exposure of internal storage and ORM fields."""
    response = upload(client)

    assert response.status_code == 201

    assert set(response.json()) == {
        "id",
        "filename",
        "sha256",
        "mime_type",
        "status",
        "created_at",
    }


def test_health_endpoint_regression(client):
    """Ensure the existing health endpoint remains available."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint_regression(client):
    """Ensure the original root endpoint remains available."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_upload_openapi_definition(client):
    """Verify the documented multipart upload contract."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    operation = response.json()["paths"][UPLOAD_URL]["post"]

    assert "multipart/form-data" in (
        operation["requestBody"]["content"]
    )

    for code in (
        "201",
        "400",
        "409",
        "413",
        "415",
        "422",
        "503",
    ):
        assert code in operation["responses"]
