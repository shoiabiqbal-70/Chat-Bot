"""Extract structured paragraphs, headings and tables from DOCX files."""

from pathlib import Path
from uuid import UUID

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

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
from app.ingestion.validation import validate_docx


DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document"
)

DEFAULT_MAX_DOCX_BLOCKS = 100_000


def _classify_paragraph(paragraph):
    """Identify headings, list items and ordinary paragraphs."""

    style_name = (
        paragraph.style.name
        if paragraph.style is not None
        else ""
    )

    if style_name.lower().startswith("heading"):
        return BlockType.HEADING

    if "list" in style_name.lower():
        return BlockType.LIST_ITEM

    return BlockType.PARAGRAPH


def _extract_table_text(table):
    """Preserve table rows and cell ordering."""

    rows = []

    for row in table.rows:
        cells = []

        for cell in row.cells:
            cell_text = "\n".join(
                paragraph.text
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            )

            cells.append(cell_text)

        rows.append(" | ".join(cells))

    return "\n".join(rows)


def extract_docx(
    path: str | Path,
    document_id: UUID,
    filename: str,
    *,
    max_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
    max_blocks: int = DEFAULT_MAX_DOCX_BLOCKS,
    max_file_bytes: int | None = None,
) -> ExtractedDocument:
    """Extract ordered text from a DOCX document.

    Preserves paragraphs, headings, lists and tables.
    DOCX pagination is not inferred.
    The original source file is never modified.
    """

    validate_positive_limit(
        max_characters,
        "Maximum extracted characters",
    )

    validate_positive_limit(
        max_blocks,
        "Maximum DOCX blocks",
    )

    if not isinstance(document_id, UUID):
        raise TypeError("Document ID must be a UUID.")

    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")

    if not filename.strip():
        raise ValueError("Filename cannot be empty.")

    source = validate_source_file(
        path,
        max_file_bytes=max_file_bytes,
    )

    blocks = []
    warnings = []
    total_characters = 0
    current_section = None

    try:
        content = source.read_bytes()
        validate_docx(content)

        document = Document(source)

        for item in document.iter_inner_content():

            if hasattr(item, "style") and hasattr(item, "text"):
                text = item.text

                if not text.strip():
                    continue

                block_type = _classify_paragraph(item)

                if block_type == BlockType.HEADING:
                    current_section = text.strip()

            else:
                text = _extract_table_text(item)

                if not text.strip():
                    continue

                block_type = BlockType.TABLE

            total_characters += len(text)

            if total_characters > max_characters:
                raise ExtractionError(
                    "DOCX exceeds the maximum extracted character limit."
                )

            if len(blocks) >= max_blocks:
                raise ExtractionError(
                    "DOCX exceeds the maximum extracted block limit."
                )

            blocks.append(
                ExtractedBlock(
                    text=text,
                    block_index=len(blocks),
                    page_number=None,
                    section=current_section,
                    block_type=block_type,
                )
            )

    except ExtractionError:
        raise

    except Exception as exc:
        raise ExtractionError(
            "Unable to read or extract the DOCX document."
        ) from exc

    if not blocks:
        warnings.append(
            "No extractable text found in the DOCX document."
        )

    return ExtractedDocument(
        document_id=document_id,
        filename=filename,
        mime_type=DOCX_MIME_TYPE,
        blocks=tuple(blocks),
        page_count=None,
        warnings=tuple(warnings),
    )
