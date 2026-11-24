
"""API endpoints for document upload and registration."""

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.ingestion.validation import UploadValidationError
from app.schemas import DocumentUploadResponse
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
    DuplicateDocumentError,
)


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
