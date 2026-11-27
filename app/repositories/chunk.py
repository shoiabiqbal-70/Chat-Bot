"""Repository layer for document chunk persistence."""

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk


class ChunkRepository:
    """Handle document chunk database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Share the caller's transaction with chunk operations."""
        self.session = session

    async def create_many(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Add multiple chunks to the current transaction."""
        if not chunks:
            return []

        self.session.add_all(chunks)
        await self.session.flush()
        return chunks

    async def get_by_id(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a chunk by its UUID."""
        result = await self.session.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()

    async def list_for_document(
        self,
        document_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """Retrieve chunks in their original document order."""
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = (
            # Preserve original chunk order for citations and whole-file scans.
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_for_document(self, document_id: UUID) -> int:
        """Count chunks belonging to a document."""
        result = await self.session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == document_id)
        )
        return result.scalar_one()

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete all chunks belonging to a document."""
        # Bulk delete bypasses ORM synchronization because the caller does not
        # need loaded chunk objects after reindexing.
        result = await self.session.execute(
            delete(Chunk)
            .where(Chunk.document_id == document_id)
            .execution_options(synchronize_session=False)
        )

        await self.session.flush()
        return result.rowcount or 0
