
"""Unit tests for local document storage."""

import hashlib
from dataclasses import replace
from uuid import uuid4

import pytest

from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.ingestion.validation import ValidatedUpload


@pytest.fixture
def storage(tmp_path):
    """Provide isolated temporary document storage."""
    return LocalDocumentStorage(tmp_path / "uploads")


@pytest.fixture
def upload():
    """Provide a validated text document."""
    content = b"Production RAG storage test."

    return ValidatedUpload(
        filename="sample.txt",
        extension=".txt",
        mime_type="text/plain",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content=content,
    )


def test_storage_directory_created(storage):
    """Create the configured storage directory."""
    assert storage.root.is_dir()


def test_save_document(storage, upload):
    """Save document content and return its metadata."""
    document_id = uuid4()

    result = storage.save(document_id, upload)

    assert result.path.is_file()
    assert result.path.read_bytes() == upload.content
    assert result.document_id == document_id
    assert result.sha256 == upload.sha256
    assert result.size_bytes == upload.size_bytes


def test_uuid_based_filename(storage, upload):
    """Use UUID filenames instead of user filenames."""
    document_id = uuid4()

    result = storage.save(document_id, upload)

    assert result.path.name == f"{document_id.hex}.txt"
    assert result.path.parent == storage.root


def test_original_filename_not_used(storage, upload):
    """Prevent user filenames from controlling storage paths."""
    unsafe_upload = replace(
        upload,
        filename="../../outside.txt",
    )

    result = storage.save(uuid4(), unsafe_upload)

    assert result.path.parent == storage.root
    assert result.path.name != "outside.txt"


def test_existing_file_not_overwritten(storage, upload):
    """Reject attempts to overwrite an existing document."""
    document_id = uuid4()

    first = storage.save(document_id, upload)

    with pytest.raises(StorageError):
        storage.save(document_id, upload)

    assert first.path.read_bytes() == upload.content


def test_distinct_document_ids(storage, upload):
    """Store separate documents under separate UUIDs."""
    first = storage.save(uuid4(), upload)
    second = storage.save(uuid4(), upload)

    assert first.path != second.path


def test_invalid_checksum(storage, upload):
    """Reject content with an incorrect checksum."""
    invalid = replace(
        upload,
        sha256="0" * 64,
    )

    with pytest.raises(StorageError):
        storage.save(uuid4(), invalid)

    assert list(storage.root.iterdir()) == []


def test_invalid_size(storage, upload):
    """Reject inconsistent document size metadata."""
    invalid = replace(upload, size_bytes=1)

    with pytest.raises(StorageError):
        storage.save(uuid4(), invalid)

    assert list(storage.root.iterdir()) == []


def test_empty_document(storage, upload):
    """Reject empty document content."""
    invalid = replace(
        upload,
        content=b"",
        size_bytes=0,
        sha256=hashlib.sha256(b"").hexdigest(),
    )

    with pytest.raises(StorageError):
        storage.save(uuid4(), invalid)


@pytest.mark.parametrize(
    "extension",
    [".exe", "../file", "/outside", ""],
)
def test_invalid_extension(storage, extension):
    """Reject unsupported or unsafe extensions."""
    with pytest.raises(ValueError):
        storage.get_path(uuid4(), extension)


def test_invalid_document_id(storage):
    """Reject identifiers that are not UUIDs."""
    with pytest.raises(ValueError):
        storage.get_path("../../outside", ".txt")


def test_document_exists(storage, upload):
    """Identify an existing stored document."""
    document_id = uuid4()

    storage.save(document_id, upload)

    assert storage.exists(document_id, ".txt")


def test_missing_document(storage):
    """Return false for a missing document."""
    assert not storage.exists(uuid4(), ".txt")


def test_delete_document(storage, upload):
    """Delete an existing stored document."""
    document_id = uuid4()

    result = storage.save(document_id, upload)

    assert storage.delete(document_id, ".txt")
    assert not result.path.exists()


def test_delete_missing_document(storage):
    """Handle deletion of a nonexistent document."""
    assert not storage.delete(uuid4(), ".txt")


def test_failed_write_cleanup(storage, upload, monkeypatch):
    """Remove a partial file when persistence fails."""

    def fail_sync(descriptor):
        """Simulate a filesystem synchronization failure."""
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(
        "app.ingestion.storage.os.fsync",
        fail_sync,
    )

    with pytest.raises(StorageError):
        storage.save(uuid4(), upload)

    assert list(storage.root.iterdir()) == []
