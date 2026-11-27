"""Extract structured text from UTF-8 TXT documents."""

import re
from pathlib import Path
from uuid import UUID

from app.ingestion.extraction import (
    DEFAULT_MAX_EXTRACTED_CHARACTERS,
    validate_positive_limit,
    validate_source_file,
)
from app.ingestion.extraction_models import (
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
)


TXT_MIME_TYPE = "text/plain"
DEFAULT_MAX_TXT_BLOCKS = 100_000


def extract_txt(
    path: str | Path,
    document_id: UUID,
    filename: str,
    *,
    max_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
    max_blocks: int = DEFAULT_MAX_TXT_BLOCKS,
    max_file_bytes: int | None = None,
) -> ExtractedDocument:
    """Extract UTF-8 text while preserving paragraph ordering."""

    validate_positive_limit(max_characters, "Maximum extracted characters")
    validate_positive_limit(max_blocks, "Maximum TXT blocks")

    if not isinstance(document_id, UUID):
        raise TypeError("Document ID must be a UUID.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid filename is required.")

    source = validate_source_file(
        path,
        max_file_bytes=max_file_bytes,
    )

    try:
        content = source.read_bytes()

        if max_file_bytes is not None and len(content) > max_file_bytes:
            raise ExtractionError("TXT exceeds the file size limit.")

        if b"\x00" in content:
            raise ExtractionError("Binary content is not allowed in TXT files.")

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ExtractionError(
                "TXT documents must use UTF-8 encoding."
            ) from None

        if not text.strip():
            raise ExtractionError("TXT document contains no readable text.")

        if len(text) > max_characters:
            raise ExtractionError(
                "TXT exceeds the maximum extracted character limit."
            )

        # Normalize line endings without changing paragraph order.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        paragraphs = re.split(r"\n[ \t]*\n+", text)
        # Blank lines separate blocks; ordinary line breaks stay inside each.

        blocks = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(blocks) >= max_blocks:
                raise ExtractionError(
                    "TXT exceeds the maximum extracted block limit."
                )

            blocks.append(
                ExtractedBlock(
                    text=paragraph,
                    block_index=len(blocks),
                    page_number=None,
                    section=None,
                    block_type=BlockType.PARAGRAPH,
                )
            )

    except ExtractionError:
        raise
    except OSError:
        raise ExtractionError(
            "Unable to access the TXT document."
        ) from None
    except Exception:
        raise ExtractionError(
            "Unable to extract the TXT document."
        ) from None

    if not blocks:
        raise ExtractionError("TXT document contains no extractable text.")

    return ExtractedDocument(
        document_id=document_id,
        filename=filename,
        mime_type=TXT_MIME_TYPE,
        blocks=tuple(blocks),
        page_count=None,
        warnings=(),
    )
