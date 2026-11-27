"""Turn a registered upload into searchable chunks."""

from pathlib import Path
import logging
from uuid import UUID, uuid4

import anyio
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunking import chunk_document
from app.ingestion.dispatcher import extract_document
from app.ingestion.embedding import get_embedding_service
from app.ingestion.extraction_models import ExtractionError
from app.ingestion.storage import LocalDocumentStorage
from app.ingestion.tokenizer import get_tokenizer
from app.models import Chunk
from app.repositories import ChunkRepository, DocumentRepository
from app.services.vector_store import VectorStore


class IngestionError(RuntimeError):
    """A registered document could not be indexed."""


class InvalidDocumentContent(IngestionError):
    """A file has no usable text or cannot be extracted."""


logger = logging.getLogger(__name__)


async def process_document(
    document_id: UUID,
    session: AsyncSession,
    storage: LocalDocumentStorage,
    vectors: VectorStore,
    *,
    reindex: bool = False,
) -> int:
    """Extract, chunk, embed, and persist a registered document.

    On failure the document remains available with a failed status for
    diagnosis. Vector upserts are compensated if the database write fails.
    """
    documents = DocumentRepository(session)
    document = await documents.get_by_id_for_update(document_id)
    if document is None:
        raise LookupError("Document not found.")
    if document.status not in {"pending", "failed"} and not (
        reindex and document.status == "completed"
    ):
        raise IngestionError("Document is not ready for processing.")

    if document.status == "failed" or reindex:
        # A prior upsert may have succeeded only in part. Clear any remnants.
        await vectors.delete_document(str(document_id))
        await ChunkRepository(session).delete_for_document(document_id)

    document.status = "processing"
    await session.commit()
    # Persist processing status before the slow extraction and embedding work.
    upsert_started = False
    try:
        path = storage.get_path(document.id, Path(document.filename).suffix)
        extracted = await anyio.to_thread.run_sync(
            lambda: extract_document(path, document.id, document.filename,
                                     mime_type=document.mime_type)
        )
        tokenizer = await anyio.to_thread.run_sync(get_tokenizer)
        parts = await anyio.to_thread.run_sync(
            lambda: chunk_document(extracted, tokenizer)
        )
        if not parts:
            raise InvalidDocumentContent("Document has no searchable text.")
        embeddings = await anyio.to_thread.run_sync(
            lambda: get_embedding_service().embed_documents([p.text for p in parts])
        )
        rows = [Chunk(id=uuid4(), document_id=document.id, text=part.text,
                      page_number=part.page_number, section=part.section,
                      chunk_index=part.chunk_index, token_count=part.token_count)
                for part in parts]
        payload = [
            # Vector metadata supports scoped search; source text stays canonical
            # in Postgres where ownership and lifecycle checks are enforced.
            {"id": str(row.id), "values": embeddings[i].tolist(),
             "metadata": {"document_id": str(document.id),
                          "chunk_id": str(row.id),
                          "filename": document.filename,
                          "chunk_index": row.chunk_index,
                          **({"owner_id": str(document.owner_id)} if getattr(document, "owner_id", None) else {})}}
            for i, row in enumerate(rows)
        ]
        upsert_started = True
        # Pinecone and Postgres cannot share a transaction, so cleanup below
        # compensates if either stage after this point fails.
        await vectors.upsert(payload)
        await ChunkRepository(session).create_many(rows)
        document.status = "completed"
        await session.commit()
        return len(rows)
    except Exception as exc:
        await session.rollback()
        # Roll back partial SQL writes before deleting any new vector records.
        cleanup_failed = False
        if upsert_started:
            try:
                await vectors.delete_document(str(document_id))
            except Exception:
                cleanup_failed = True
                logger.exception("Vector cleanup failed for document %s", document_id)
        document.status = "failed"
        document.error_message = (
            "Vector cleanup failed; retry required."
            if cleanup_failed else type(exc).__name__
        )
        await session.commit()
        if isinstance(exc, (ExtractionError, InvalidDocumentContent)):
            raise InvalidDocumentContent("Document has no extractable text.") from exc
        raise IngestionError("Document indexing failed.") from exc
