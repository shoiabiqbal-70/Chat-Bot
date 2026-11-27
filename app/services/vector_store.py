"""Pinecone vector operations isolated from document processing."""

import anyio
from pinecone import Pinecone
from pinecone.errors.exceptions import (
    PineconeConnectionError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)

from app.config import settings


class VectorStoreError(RuntimeError):
    """A vector database operation failed."""


class VectorStore:
    """Use a configured Pinecone dense index with cosine scores."""

    def __init__(self, index=None):
        """Accept an injected index for tests or connect to the configured one."""
        if index is None:
            secret = settings.pinecone_api_key
            if secret is None or not secret.get_secret_value().strip():
                raise VectorStoreError("PINECONE_API_KEY is not configured.")
            try:
                index = Pinecone(api_key=secret.get_secret_value()).Index(
                    settings.pinecone_index_name
                )
            except Exception as exc:
                raise VectorStoreError("Vector index is unavailable.") from exc
        self.index = index

    async def _execute(self, operation):
        """Retry transient Pinecone faults with bounded backoff."""
        for attempt in range(3):
            try:
                return await anyio.to_thread.run_sync(operation)
            except (PineconeConnectionError, PineconeTimeoutError,
                    RateLimitError, ServiceError):
                if attempt == 2:
                    raise
                await anyio.sleep(2 ** attempt)

    async def upsert(self, vectors: list[dict]) -> None:
        """Write vectors without blocking the event loop."""
        try:
            # Small batches limit request size and memory during large uploads.
            for start in range(0, len(vectors), 100):
                batch = vectors[start:start + 100]
                await self._execute(lambda: self.index.upsert(vectors=batch))
        except Exception as exc:
            raise VectorStoreError("Vector upsert failed.") from exc

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a document."""
        try:
            await self._execute(
                lambda: self.index.delete(filter={"document_id": document_id})
            )
        except Exception as exc:
            raise VectorStoreError("Vector deletion failed.") from exc

    async def query(self, vector: list[float], top_k: int,
                    document_id: str | None = None,
                    owner_id: str | None = None) -> list:
        """Return scored vector matches."""
        try:
            options = {"vector": vector, "top_k": top_k, "include_metadata": True}
            filters = {}
            if document_id is not None:
                filters["document_id"] = {"$eq": document_id}
            if owner_id is not None:
                # Scope retrieval at Pinecone before checking ownership again
                # against canonical document rows in the answer service.
                filters["owner_id"] = {"$eq": owner_id}
            if filters:
                options["filter"] = filters
            response = await self._execute(lambda: self.index.query(**options))
            return list(response.matches or [])
        except Exception as exc:
            raise VectorStoreError("Vector query failed.") from exc
