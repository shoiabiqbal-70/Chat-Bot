"""Repository layer for document persistence."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


class DocumentRepository:
    """Handle document database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, document: Document) -> Document:
        """Add a document to the current transaction."""
        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """Retrieve a document by its UUID."""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_update(self, document_id: UUID) -> Document | None:
        """Lock one row while claiming a document for processing."""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_by_sha256(self, sha256: str, owner_id: UUID | None = None) -> Document | None:
        """Retrieve a document by its SHA-256 digest."""
        result = await self.session.execute(
            select(Document).where(Document.sha256 == sha256,
                                   Document.owner_id == owner_id if owner_id is not None else Document.owner_id.is_(None))
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        owner_id: UUID | None = None,
    ) -> list[Document]:
        """Return documents with optional status filtering."""
        if limit < 1 or limit > 500:
            raise ValueError("Limit must be between 1 and 500.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = select(Document)

        if owner_id is not None:
            statement = statement.where(Document.owner_id == owner_id)

        if status is not None:
            statement = statement.where(Document.status == status)

        statement = (
            statement
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update_status(
        self,
        document_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> Document | None:
        """Update document processing status."""
        document = await self.get_by_id(document_id)

        if document is None:
            return None

        document.status = status
        document.error_message = error_message

        await self.session.flush()
        return document

    async def delete(self, document_id: UUID) -> bool:
        """Delete a document and its associated chunks."""
        document = await self.get_by_id(document_id)

        if document is None:
            return False

        await self.session.delete(document)
        await self.session.flush()
        return True
