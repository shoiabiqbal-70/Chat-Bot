
"""API endpoints for document upload and registration."""

import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID

import anyio

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_db
from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.ingestion.validation import UploadValidationError
from app.schemas import DocumentIngestResponse, DocumentUploadResponse
from app.repositories import DocumentRepository
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
    DuplicateDocumentError,
)
from app.services.ingestion_service import IngestionError, InvalidDocumentContent, process_document
from app.services.vector_store import VectorStore, VectorStoreError


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def get_document_storage() -> LocalDocumentStorage:
    """Provide the local document storage dependency."""
    try:
        return LocalDocumentStorage()

    except StorageError as exc:
        logger.error("Document storage initialization failed.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is unavailable.",
        ) from exc


@router.get("", response_model=list[DocumentUploadResponse])
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentUploadResponse]:
    """List registered documents in newest-first order."""
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(status_code=422, detail="Invalid pagination values.")
    try:
        documents = await DocumentRepository(session).list_documents(limit, offset)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Document database is unavailable.") from exc
    return [DocumentUploadResponse.model_validate(item) for item in documents]


@router.get("/{document_id}", response_model=DocumentUploadResponse)
async def get_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentUploadResponse:
    """Read registration status and metadata for a document."""
    try:
        document = await DocumentRepository(session).get_by_id(document_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Document database is unavailable.") from exc
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentUploadResponse.model_validate(document)


@router.post("/{document_id}/process", status_code=200)
async def process_registered_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[LocalDocumentStorage, Depends(get_document_storage)],
) -> dict[str, int]:
    """Index a pending or failed registered document for retrieval."""
    try:
        count = await process_document(document_id, session, storage, VectorStore())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidDocumentContent as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (IngestionError, VectorStoreError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc
    return {"chunk_count": count}


@router.post("/{document_id}/reindex", status_code=200)
async def reindex_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[LocalDocumentStorage, Depends(get_document_storage)],
) -> dict[str, int]:
    """Replace existing vectors and chunks with the current chunk settings."""
    try:
        count = await process_document(document_id, session, storage,
                                       VectorStore(), reindex=True)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidDocumentContent as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (IngestionError, VectorStoreError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Document reindexing failed.") from exc
    return {"chunk_count": count}


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[LocalDocumentStorage, Depends(get_document_storage)],
) -> None:
    """Remove vectors before deleting canonical records and local content."""
    repository = DocumentRepository(session)
    try:
        document = await repository.get_by_id(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        extension = Path(document.filename).suffix
        document.status = "deleting"
        await session.commit()
        await VectorStore().delete_document(str(document_id))
        await repository.delete(document_id)
        await session.commit()
        await anyio.to_thread.run_sync(storage.delete, document_id, extension)
    except HTTPException:
        raise
    except (SQLAlchemyError, VectorStoreError, StorageError) as exc:
        await session.rollback()
        raise HTTPException(status_code=503, detail="Document deletion failed.") from exc


@router.post("", response_model=DocumentIngestResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Upload and index a document")
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX or UTF-8 TXT document.")],
    session: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[LocalDocumentStorage, Depends(get_document_storage)],
) -> DocumentIngestResponse:
    """Return 201 only after the uploaded file is searchable."""
    registered = await upload_document(file, session, storage)
    try:
        count = await process_document(registered.id, session, storage, VectorStore())
    except InvalidDocumentContent as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (IngestionError, VectorStoreError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc
    document = await DocumentRepository(session).get_by_id(registered.id)
    return DocumentIngestResponse.model_validate(
        {**DocumentUploadResponse.model_validate(document).model_dump(),
         "chunk_count": count}
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    responses={
        400: {"description": "Invalid or empty document"},
        409: {"description": "Duplicate document"},
        413: {"description": "Document exceeds size limit"},
        415: {"description": "Unsupported document format"},
        503: {"description": "Registration service unavailable"},
    },
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="PDF, DOCX or UTF-8 TXT document."),
    ],
    session: Annotated[
        AsyncSession,
        Depends(get_db),
    ],
    storage: Annotated[
        LocalDocumentStorage,
        Depends(get_document_storage),
    ],
) -> DocumentUploadResponse:
    """Validate, store and register an uploaded document."""

    service = DocumentService(
        session=session,
        storage=storage,
    )

    try:
        document = await service.register(file)

        return DocumentUploadResponse.model_validate(
            document
        )

    except UploadValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except DocumentServiceError as exc:
        logger.error("Document registration failed.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document registration is unavailable.",
        ) from exc

    finally:
        await file.close()
