"""Pinecone adapter retry and batching behavior."""

import asyncio
from types import SimpleNamespace

import pytest
from pinecone.errors.exceptions import PineconeConnectionError
from pydantic import SecretStr

from app.services import vector_store


def test_index_connection_failure_is_service_error(monkeypatch):
    def unavailable(**kwargs):
        raise PineconeConnectionError("Connection refused")

    monkeypatch.setattr(vector_store, "settings", SimpleNamespace(
        pinecone_api_key=SecretStr("test-key"), pinecone_index_name="test-index"
    ))
    monkeypatch.setattr(vector_store, "Pinecone", unavailable)
    with pytest.raises(vector_store.VectorStoreError, match="unavailable"):
        vector_store.VectorStore()


def test_transient_upsert_is_retried(monkeypatch):
    class Index:
        def __init__(self):
            self.calls = 0

        def upsert(self, vectors):
            self.calls += 1
            if self.calls == 1:
                raise PineconeConnectionError("Temporary connection failure")

    async def no_delay(seconds):
        pass

    monkeypatch.setattr(vector_store.anyio, "sleep", no_delay)
    index = Index()
    asyncio.run(vector_store.VectorStore(index).upsert([{"id": "one"}]))
    assert index.calls == 2


def test_upsert_is_batched():
    class Index:
        def __init__(self):
            self.batch_sizes = []

        def upsert(self, vectors):
            self.batch_sizes.append(len(vectors))

    index = Index()
    asyncio.run(vector_store.VectorStore(index).upsert(
        [{"id": str(i)} for i in range(205)]
    ))
    assert index.batch_sizes == [100, 100, 5]


def test_query_filters_selected_document():
    class Index:
        def query(self, **kwargs):
            assert kwargs["filter"] == {"document_id": {"$eq": "selected-id"}}
            return SimpleNamespace(matches=[])

    result = asyncio.run(vector_store.VectorStore(Index()).query([1.0], 5, "selected-id"))
    assert result == []
