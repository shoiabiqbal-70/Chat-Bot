"""Dispatch supported documents to their respective extractors."""

from pathlib import Path
from uuid import UUID

from app.ingestion.docx_extraction import (
    DOCX_MIME_TYPE,
    extract_docx,
)
from app.ingestion.extraction import (
    PDF_MIME_TYPE,
    extract_pdf,
)
from app.ingestion.extraction_models import (
    ExtractedDocument,
    ExtractionError,
)
from app.ingestion.txt_extraction import (
    TXT_MIME_TYPE,
    extract_txt,
)


SUPPORTED_FORMATS = {
    ".pdf": PDF_MIME_TYPE,
    ".docx": DOCX_MIME_TYPE,
    ".txt": TXT_MIME_TYPE,
}


def extract_document(
    path: str | Path,
    document_id: UUID,
    filename: str,
    *,
    mime_type: str | None = None,
    max_pages: int = 500,
    max_characters: int = 2_000_000,
    max_blocks: int = 100_000,
    max_file_bytes: int | None = None,
) -> ExtractedDocument:
    """Extract a supported document using its appropriate extractor."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid filename is required.")

    if not isinstance(document_id, UUID):
        raise TypeError("Document ID must be a UUID.")

    extension = Path(filename).suffix.lower()

    expected_mime = SUPPORTED_FORMATS.get(extension)

    if expected_mime is None:
        raise ExtractionError(
            "Unsupported document format. Use PDF, DOCX or TXT."
        )

    if mime_type is not None and mime_type != expected_mime:
        raise ExtractionError(
            "Document extension and MIME type do not match."
        )

    if extension == ".pdf":
        return extract_pdf(
            path,
            document_id,
            filename,
            max_pages=max_pages,
            max_characters=max_characters,
            max_file_bytes=max_file_bytes,
        )

    if extension == ".docx":
        return extract_docx(
            path,
            document_id,
            filename,
            max_characters=max_characters,
            max_blocks=max_blocks,
            max_file_bytes=max_file_bytes,
        )

    if extension == ".txt":
        return extract_txt(
            path,
            document_id,
            filename,
            max_characters=max_characters,
            max_blocks=max_blocks,
            max_file_bytes=max_file_bytes,
        )

    raise ExtractionError("Unsupported document format.")
