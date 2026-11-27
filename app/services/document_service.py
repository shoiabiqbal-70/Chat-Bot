
"""Coordinate document validation, storage and database registration."""

import logging
from uuid import UUID, uuid4

import anyio
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.storage import (
    LocalDocumentStorage,
    StorageError,
)
from app.ingestion.validation import validate_upload
from app.models import Document
from app.repositories import DocumentRepository


logger = logging.getLogger(__name__)


class DocumentServiceError(RuntimeError):
    """Represent a document registration failure."""


class DuplicateDocumentError(DocumentServiceError):
    """Represent an upload whose content already exists."""


def is_duplicate_hash_error(error: IntegrityError) -> bool:
    """Identify PostgreSQL document hash uniqueness violations."""
    original = error.orig

    for candidate in (
        original,
        getattr(original, "__cause__", None),
    ):
        if candidate is None:
            continue

        constraint = getattr(
            candidate,
            "constraint_name",
            None,
        )

        if constraint in {"uq_documents_sha256", "uq_documents_owner_sha256"}:
            return True

    return False


class DocumentService:
    """Coordinate document registration operations."""

    def __init__(
        self,
        session: AsyncSession,
        storage: LocalDocumentStorage,
        owner_id: UUID | None = None,
    ) -> None:
        """Initialize service dependencies."""
        self.session = session
        self.storage = storage
        self.repository = DocumentRepository(session)
        self.owner_id = owner_id

    async def register(
        self,
        upload: UploadFile,
    ) -> Document:
        """Validate, store and register a new document."""

        validated = await validate_upload(upload)

        document_id = uuid4()

        document = Document(
            id=document_id,
            owner_id=self.owner_id,
            filename=validated.filename,
            sha256=validated.sha256,
            mime_type=validated.mime_type,
            status="pending",
        )

        stored = False
        commit_started = False

        if self.session.in_transaction():
            raise DocumentServiceError(
                "Document registration requires a fresh transaction."
            )

        # Keep transaction finalization protected from
        # request cancellation.
        with anyio.CancelScope(shield=True):
            try:
                existing = await self.repository.get_by_sha256(validated.sha256, self.owner_id)

                if existing is not None:
                    raise DuplicateDocumentError(
                        "A document with identical content already exists."
                    )

                await self.repository.create(document)

                await anyio.to_thread.run_sync(
                    self.storage.save,
                    document_id,
                    validated,
                )

                stored = True
                commit_started = True

                await self.session.commit()

                return document

            except DuplicateDocumentError:
                await self.session.rollback()
                raise

            except IntegrityError as exc:
                await self.session.rollback()

                if stored and not commit_started:
                    await self._cleanup(
                        document_id,
                        validated.extension,
                    )

                if is_duplicate_hash_error(exc):
                    raise DuplicateDocumentError(
                        "A document with identical content already exists."
                    ) from exc

                raise DocumentServiceError(
                    "Document metadata could not be registered."
                ) from exc

            except (SQLAlchemyError, StorageError) as exc:
                await self.session.rollback()

                if stored and not commit_started:
                    await self._cleanup(
                        document_id,
                        validated.extension,
                    )

                if commit_started:
                    logger.error(
                        "Uncertain registration outcome for document %s",
                        document_id,
                    )

                raise DocumentServiceError(
                    "Document registration failed."
                ) from exc

            except Exception:
                await self.session.rollback()

                if stored and not commit_started:
                    await self._cleanup(
                        document_id,
                        validated.extension,
                    )

                raise

    async def _cleanup(
        self,
        document_id,
        extension: str,
    ) -> None:
        """Attempt to remove a file after registration failure."""
        try:
            await anyio.to_thread.run_sync(
                self.storage.delete,
                document_id,
                extension,
            )

        except StorageError:
            logger.error(
                "Storage cleanup failed for document %s",
                document_id,
            )
