"""Failure handling for the indexing workflow."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import numpy as np

from app.ingestion.extraction_models import ExtractionError
from app.services import ingestion_service


def test_unextractable_document_is_failed(monkeypatch, tmp_path):
    document = SimpleNamespace(id=uuid4(), filename="empty.pdf",
                               mime_type="application/pdf", status="pending",
                               error_message=None)
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())

    class Repository:
        def __init__(self, db):
            pass

        async def get_by_id_for_update(self, document_id):
            return document

    class Storage:
        def get_path(self, document_id, extension):
            return tmp_path / "empty.pdf"

    class Vectors:
        async def delete_document(self, document_id):
            raise AssertionError("No vectors should exist before upsert")

    def fail_extract(*args, **kwargs):
        raise ExtractionError("No text")

    monkeypatch.setattr(ingestion_service, "DocumentRepository", Repository)
    monkeypatch.setattr(ingestion_service, "extract_document", fail_extract)
    with pytest.raises(ingestion_service.InvalidDocumentContent):
        asyncio.run(ingestion_service.process_document(
            document.id, session, Storage(), Vectors()
        ))
    assert document.status == "failed"
    assert document.error_message == "ExtractionError"
    assert session.commit.await_count == 2


def test_partial_vector_upsert_is_compensated(monkeypatch, tmp_path):
    document = SimpleNamespace(id=uuid4(), filename="sample.pdf",
                               mime_type="application/pdf", status="pending",
                               error_message=None)
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())

    class Repository:
        def __init__(self, db):
            pass

        async def get_by_id_for_update(self, document_id):
            return document

    class Storage:
        def get_path(self, document_id, extension):
            return tmp_path / "sample.pdf"

    class Vectors:
        def __init__(self):
            self.deleted = []

        async def upsert(self, payload):
            raise RuntimeError("Simulated partial write")

        async def delete_document(self, document_id):
            self.deleted.append(document_id)

    class Embedding:
        def embed_documents(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    part = SimpleNamespace(text="Useful text", page_number=1, section=None,
                           chunk_index=0, token_count=2)
    monkeypatch.setattr(ingestion_service, "DocumentRepository", Repository)
    monkeypatch.setattr(ingestion_service, "extract_document", lambda *a, **k: object())
    monkeypatch.setattr(ingestion_service, "get_tokenizer", lambda: object())
    monkeypatch.setattr(ingestion_service, "chunk_document", lambda *a: [part])
    monkeypatch.setattr(ingestion_service, "get_embedding_service", lambda: Embedding())
    vectors = Vectors()
    with pytest.raises(ingestion_service.IngestionError):
        asyncio.run(ingestion_service.process_document(
            document.id, session, Storage(), vectors
        ))
    assert vectors.deleted == [str(document.id)]
    assert document.status == "failed"


def test_reindex_replaces_old_vectors_before_extraction(monkeypatch, tmp_path):
    document = SimpleNamespace(id=uuid4(), filename="sample.pdf",
                               mime_type="application/pdf", status="completed",
                               error_message=None)
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    events = []

    class Repository:
        def __init__(self, db):
            pass

        async def get_by_id_for_update(self, document_id):
            return document

    class Chunks:
        def __init__(self, db):
            pass

        async def delete_for_document(self, document_id):
            events.append("chunks")

    class Storage:
        def get_path(self, document_id, extension):
            return tmp_path / "sample.pdf"

    class Vectors:
        async def delete_document(self, document_id):
            events.append("vectors")

    def fail_extract(*args, **kwargs):
        events.append("extract")
        raise ExtractionError("No text")

    monkeypatch.setattr(ingestion_service, "DocumentRepository", Repository)
    monkeypatch.setattr(ingestion_service, "ChunkRepository", Chunks)
    monkeypatch.setattr(ingestion_service, "extract_document", fail_extract)
    with pytest.raises(ingestion_service.InvalidDocumentContent):
        asyncio.run(ingestion_service.process_document(
            document.id, session, Storage(), Vectors(), reindex=True
        ))
    assert events == ["vectors", "chunks", "extract"]
    assert document.status == "failed"
