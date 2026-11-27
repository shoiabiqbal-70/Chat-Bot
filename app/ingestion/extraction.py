
"""Extract structured text and citation metadata from documents."""

from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from app.config import settings
from app.ingestion.extraction_models import (
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
)


PDF_MIME_TYPE = "application/pdf"

DEFAULT_MAX_PDF_PAGES = 500
DEFAULT_MAX_EXTRACTED_CHARACTERS = 2_000_000


def validate_positive_limit(
    value: int,
    name: str,
) -> None:
    """Require a strictly positive integer resource limit."""

    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer."
        )


def validate_source_file(
    path: str | Path,
    max_file_bytes: int | None = None,
) -> Path:
    """Validate the source file before document extraction."""

    if max_file_bytes is None:
        max_file_bytes = (
            settings.max_upload_size_mb * 1024 * 1024
        )

    validate_positive_limit(
        max_file_bytes,
        "Maximum file size",
    )

    source = Path(path)

    try:
        if source.is_symlink():
            raise ExtractionError(
                "Symbolic links are not supported."
            )

        if not source.is_file():
            raise ExtractionError(
                "Source document does not exist "
                "or is not a regular file."
            )

        size_bytes = source.stat().st_size

        if size_bytes == 0:
            raise ExtractionError(
                "Source document is empty."
            )

        if size_bytes > max_file_bytes:
            raise ExtractionError(
                "Source document exceeds the file size limit."
            )

    except OSError as exc:
        raise ExtractionError(
            "Unable to access the source document."
        ) from exc

    return source


def extract_pdf(
    path: str | Path,
    document_id: UUID,
    filename: str,
    *,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
    max_file_bytes: int | None = None,
) -> ExtractedDocument:
    """Extract PDF text while preserving original page numbers.

    Empty pages are skipped and recorded as warnings.

    Encrypted documents, invalid PDF files and documents
    exceeding configured extraction limits are rejected.

    This function does not perform optical character
    recognition or modify the original source document.
    """

    validate_positive_limit(
        max_pages,
        "Maximum PDF pages",
    )

    validate_positive_limit(
        max_characters,
        "Maximum extracted characters",
    )

    if not isinstance(document_id, UUID):
        raise TypeError(
            "Document ID must be a UUID."
        )

    if not isinstance(filename, str):
        raise TypeError(
            "Filename must be a string."
        )

    if not filename.strip():
        raise ValueError(
            "Filename cannot be empty."
        )

    source = validate_source_file(
        path,
        max_file_bytes=max_file_bytes,
    )

    blocks: list[ExtractedBlock] = []
    warnings: list[str] = []

    total_characters = 0

    try:
        with source.open("rb") as stream:

            if stream.read(5) != b"%PDF-":
                raise ExtractionError(
                    "Source document is not a valid PDF."
                )

            stream.seek(0)

            reader = PdfReader(
                stream,
                strict=False,
            )

            if reader.is_encrypted:
                raise ExtractionError(
                    "Encrypted PDF documents are not supported."
                )

            page_count = len(reader.pages)

            if page_count > max_pages:
                raise ExtractionError(
                    "PDF exceeds the maximum page limit."
                )

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                try:
                    page_text = page.extract_text() or ""

                except Exception as exc:
                    raise ExtractionError(
                        f"Unable to extract text from "
                        f"PDF page {page_number}."
                    ) from exc

                if not page_text.strip():
                    warnings.append(
                        f"Page {page_number} contains "
                        "no extractable text."
                    )

                    continue

                total_characters += len(page_text)

                if total_characters > max_characters:
                    raise ExtractionError(
                        "PDF exceeds the maximum extracted "
                        "character limit."
                    )

                blocks.append(
                    ExtractedBlock(
                        text=page_text,
                        block_index=len(blocks),
                        page_number=page_number,
                        section=None,
                        block_type=BlockType.TEXT,
                    )
                )

    except ExtractionError:
        raise

    except Exception as exc:
        raise ExtractionError(
            "Unable to read or extract the PDF document."
        ) from exc

    if not blocks:
        warnings.append(
            "No extractable text found. "
            "The PDF may require OCR."
        )

    return ExtractedDocument(
        document_id=document_id,
        filename=filename,
        mime_type=PDF_MIME_TYPE,
        blocks=tuple(blocks),
        page_count=page_count,
        warnings=tuple(warnings),
    )
