
"""Structured data models for document text extraction."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.ingestion.validation import ALLOWED_MIME_TYPES


class ExtractionError(RuntimeError):
    """Represent a document text extraction failure."""


class BlockType(StrEnum):
    """Identify the semantic type of an extracted text block."""

    TEXT = "text"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST_ITEM = "list_item"


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    """Represent an ordered text block with citation metadata."""

    text: str
    block_index: int
    page_number: int | None = None
    section: str | None = None
    block_type: BlockType = BlockType.TEXT

    def __post_init__(self) -> None:
        """Validate extracted text and its source metadata."""
        # Validate at the boundary so later chunking can trust provenance.

        if not isinstance(self.text, str):
            raise TypeError("Block text must be a string.")

        if not self.text.strip():
            raise ValueError("Block text cannot be empty.")

        if type(self.block_index) is not int:
            raise TypeError("Block index must be an integer.")

        if self.block_index < 0:
            raise ValueError("Block index cannot be negative.")

        if self.page_number is not None:
            if type(self.page_number) is not int:
                raise TypeError(
                    "Page number must be an integer."
                )

            if self.page_number < 1:
                raise ValueError(
                    "Page number must be at least one."
                )

        if self.section is not None:
            if not isinstance(self.section, str):
                raise TypeError(
                    "Section must be a string."
                )

            if not self.section.strip():
                raise ValueError(
                    "Section cannot be empty."
                )

        if not isinstance(self.block_type, BlockType):
            raise TypeError(
                "Block type must be a BlockType value."
            )

    @property
    def character_count(self) -> int:
        """Return the number of characters in this block."""

        return len(self.text)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Represent structured content extracted from a document."""

    document_id: UUID
    filename: str
    mime_type: str
    blocks: tuple[ExtractedBlock, ...]
    page_count: int | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate document metadata and block consistency."""
        # This is the format-independent contract shared by all extractors.

        if not isinstance(self.document_id, UUID):
            raise TypeError(
                "Document ID must be a UUID."
            )

        if not isinstance(self.filename, str):
            raise TypeError(
                "Filename must be a string."
            )

        if not self.filename.strip():
            raise ValueError(
                "Filename cannot be empty."
            )

        if self.mime_type not in ALLOWED_MIME_TYPES.values():
            raise ValueError(
                "Unsupported document MIME type."
            )

        if not isinstance(self.blocks, tuple):
            raise TypeError(
                "Document blocks must be a tuple."
            )

        if not isinstance(self.warnings, tuple):
            raise TypeError(
                "Extraction warnings must be a tuple."
            )

        for warning in self.warnings:
            if not isinstance(warning, str):
                raise TypeError(
                    "Extraction warnings must be strings."
                )

            if not warning.strip():
                raise ValueError(
                    "Extraction warnings cannot be empty."
                )

        if self.page_count is not None:
            if type(self.page_count) is not int:
                raise TypeError(
                    "Page count must be an integer."
                )

            if self.page_count < 0:
                raise ValueError(
                    "Page count cannot be negative."
                )

        previous_page = 0

        for expected_index, block in enumerate(self.blocks):

            if not isinstance(block, ExtractedBlock):
                raise TypeError(
                    "Invalid extracted block."
                )

            if block.block_index != expected_index:
                raise ValueError(
                    "Block indexes must be sequential."
                )

            if block.page_number is not None:

                if (
                    self.page_count is not None
                    and block.page_number > self.page_count
                ):
                    raise ValueError(
                        "Block page exceeds document page count."
                    )

                if block.page_number < previous_page:
                    raise ValueError(
                        "Block pages must be in document order."
                    )

                previous_page = block.page_number

    @property
    def block_count(self) -> int:
        """Return the total number of extracted blocks."""

        return len(self.blocks)

    @property
    def character_count(self) -> int:
        """Return the total number of extracted characters."""

        return sum(
            block.character_count
            for block in self.blocks
        )

    @property
    def has_text(self) -> bool:
        """Indicate whether any usable text was extracted."""

        return bool(self.blocks)

    @property
    def full_text(self) -> str:
        """Combine extracted blocks in their original order."""

        return "\n\n".join(
            block.text
            for block in self.blocks
        )
