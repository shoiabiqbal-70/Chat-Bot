
"""Unit tests for document extraction and citation metadata."""

import hashlib
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from app.ingestion.extraction import (
    extract_pdf,
    validate_positive_limit,
    validate_source_file,
)

from app.ingestion.extraction_models import (
    BlockType,
    ExtractionError,
)


# ==========================================
# PDF test fixtures
# ==========================================


def create_pdf(
    path: Path,
    page_texts: list[str | None],
    encrypted: bool = False,
) -> Path:
    """Create a genuine PDF with optional text and encryption."""

    writer = PdfWriter()

    for text in page_texts:

        page = writer.add_blank_page(
            width=612,
            height=792,
        )

        if text is None:
            continue

        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })

        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): font,
            }),
        })

        escaped = (
            text.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
        )

        content = (
            "BT /F1 12 Tf 72 720 Td "
            f"({escaped}) Tj ET"
        )

        stream = DecodedStreamObject()
        stream.set_data(content.encode("ascii"))

        page[NameObject("/Contents")] = stream

    if encrypted:
        writer.encrypt("test-password")

    with path.open("wb") as output:
        writer.write(output)

    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a three-page PDF containing one blank page."""

    return create_pdf(
        tmp_path / "sample.pdf",
        [
            "Introduction to Production RAG.",
            None,
            "PostgreSQL stores canonical chunks.",
        ],
    )


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    """Create a single-page PDF containing readable text."""

    return create_pdf(
        tmp_path / "single.pdf",
        ["Document extraction test."],
    )


# ==========================================
# Resource limit validation
# ==========================================


@pytest.mark.parametrize("value", [1, 100, 500])
def test_valid_positive_limits(value):
    """Accept positive integer resource limits."""

    validate_positive_limit(value, "Test limit")


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, 1.5, "10", None],
)
def test_invalid_positive_limits(value):
    """Reject invalid resource limits."""

    with pytest.raises(ValueError, match="positive integer"):
        validate_positive_limit(value, "Test limit")


# ==========================================
# Source file validation
# ==========================================


def test_valid_source_file(sample_pdf):
    """Accept an existing non-empty source document."""

    assert validate_source_file(sample_pdf) == sample_pdf


def test_source_file_string_path(sample_pdf):
    """Accept a source path supplied as a string."""

    assert validate_source_file(str(sample_pdf)) == sample_pdf


def test_missing_source_file(tmp_path):
    """Reject nonexistent source documents."""

    with pytest.raises(ExtractionError, match="does not exist"):
        validate_source_file(tmp_path / "missing.pdf")


def test_directory_source_rejection(tmp_path):
    """Reject directories supplied as source documents."""

    with pytest.raises(ExtractionError, match="regular file"):
        validate_source_file(tmp_path)


def test_empty_source_file(tmp_path):
    """Reject empty source documents."""

    source = tmp_path / "empty.pdf"
    source.touch()

    with pytest.raises(ExtractionError, match="empty"):
        validate_source_file(source)


def test_source_file_size_limit(sample_pdf):
    """Reject documents exceeding the source file size limit."""

    with pytest.raises(ExtractionError, match="file size limit"):
        validate_source_file(
            sample_pdf,
            max_file_bytes=10,
        )


def test_exact_source_file_size_limit(sample_pdf):
    """Accept documents exactly at the source file size limit."""

    size = sample_pdf.stat().st_size

    assert validate_source_file(
        sample_pdf,
        max_file_bytes=size,
    ) == sample_pdf


def test_source_symbolic_link_rejection(tmp_path, sample_pdf):
    """Reject symbolic links when supported by the operating system."""

    link = tmp_path / "linked.pdf"

    try:
        link.symlink_to(sample_pdf)

    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links unavailable in this environment.")

    with pytest.raises(ExtractionError, match="Symbolic links"):
        validate_source_file(link)


# ==========================================
# Successful PDF extraction
# ==========================================


def test_pdf_document_metadata(sample_pdf):
    """Preserve document identity and MIME metadata."""

    document_id = uuid4()

    result = extract_pdf(
        sample_pdf,
        document_id,
        "original-name.pdf",
    )

    assert result.document_id == document_id
    assert result.filename == "original-name.pdf"
    assert result.mime_type == "application/pdf"
    assert result.page_count == 3


def test_real_pdf_text_extraction(sample_pdf):
    """Extract actual text from PDF content streams."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert result.has_text
    assert result.block_count == 2

    assert result.blocks[0].text.strip() == (
        "Introduction to Production RAG."
    )

    assert result.blocks[1].text.strip() == (
        "PostgreSQL stores canonical chunks."
    )


def test_original_page_numbers(sample_pdf):
    """Preserve original page numbers across blank pages."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert [
        block.page_number
        for block in result.blocks
    ] == [1, 3]


def test_sequential_block_indexes(sample_pdf):
    """Assign sequential block indexes independently of pages."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert [
        block.block_index
        for block in result.blocks
    ] == [0, 1]


def test_pdf_block_metadata(sample_pdf):
    """Assign correct semantic types without invented sections."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert all(
        block.block_type is BlockType.TEXT
        for block in result.blocks
    )

    assert all(
        block.section is None
        for block in result.blocks
    )


def test_blank_page_warning(sample_pdf):
    """Record the original number of each blank page."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert result.warnings == (
        "Page 2 contains no extractable text.",
    )


def test_single_page_pdf(single_page_pdf):
    """Extract a single-page document without warnings."""

    result = extract_pdf(
        single_page_pdf,
        uuid4(),
        "single.pdf",
    )

    assert result.page_count == 1
    assert result.block_count == 1
    assert result.blocks[0].page_number == 1
    assert result.warnings == ()


def test_pdf_character_count(sample_pdf):
    """Calculate the character count across extracted blocks."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert result.character_count == sum(
        len(block.text)
        for block in result.blocks
    )


def test_pdf_full_text(sample_pdf):
    """Reconstruct extracted text in its original page order."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    assert result.full_text == "\n\n".join(
        block.text
        for block in result.blocks
    )


def test_pdf_source_preservation(sample_pdf):
    """Ensure extraction does not modify the original PDF."""

    original_hash = hashlib.sha256(
        sample_pdf.read_bytes()
    ).hexdigest()

    extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
    )

    final_hash = hashlib.sha256(
        sample_pdf.read_bytes()
    ).hexdigest()

    assert original_hash == final_hash


def test_deterministic_pdf_extraction(sample_pdf):
    """Return identical structured content for repeated extraction."""

    document_id = uuid4()

    first = extract_pdf(
        sample_pdf,
        document_id,
        "sample.pdf",
    )

    second = extract_pdf(
        sample_pdf,
        document_id,
        "sample.pdf",
    )

    assert first == second


def test_pdf_special_characters(tmp_path):
    """Preserve escaped parentheses and backslashes."""

    text = r"RAG (retrieval) uses path C:\documents."

    source = create_pdf(
        tmp_path / "special.pdf",
        [text],
    )

    result = extract_pdf(
        source,
        uuid4(),
        "special.pdf",
    )

    assert result.blocks[0].text.strip() == text


# ==========================================
# Empty and unsupported PDFs
# ==========================================


def test_all_blank_pdf(tmp_path):
    """Represent PDFs containing no extractable text."""

    source = create_pdf(
        tmp_path / "blank.pdf",
        [None, None],
    )

    result = extract_pdf(
        source,
        uuid4(),
        "blank.pdf",
    )

    assert result.page_count == 2
    assert result.block_count == 0
    assert result.has_text is False
    assert result.full_text == ""
    assert len(result.warnings) == 3

    assert "OCR" in result.warnings[-1]


def test_encrypted_pdf_rejection(tmp_path):
    """Reject password-protected PDF documents."""

    source = create_pdf(
        tmp_path / "encrypted.pdf",
        ["Protected content."],
        encrypted=True,
    )

    with pytest.raises(ExtractionError, match="Encrypted PDF"):
        extract_pdf(
            source,
            uuid4(),
            "encrypted.pdf",
        )


def test_invalid_pdf_signature(tmp_path):
    """Reject non-PDF content with a PDF filename."""

    source = tmp_path / "invalid.pdf"
    source.write_bytes(b"This is not a PDF.")

    with pytest.raises(ExtractionError, match="not a valid PDF"):
        extract_pdf(
            source,
            uuid4(),
            "invalid.pdf",
        )


def test_corrupted_pdf_structure(tmp_path):
    """Reject corrupted PDFs with a valid initial signature."""

    source = tmp_path / "corrupted.pdf"

    source.write_bytes(
        b"%PDF-1.7\nInvalid document structure"
    )

    with pytest.raises(ExtractionError, match="Unable to read"):
        extract_pdf(
            source,
            uuid4(),
            "corrupted.pdf",
        )


# ==========================================
# Extraction limit enforcement
# ==========================================


def test_pdf_page_limit(sample_pdf):
    """Reject PDFs exceeding the maximum page count."""

    with pytest.raises(ExtractionError, match="maximum page limit"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            "sample.pdf",
            max_pages=2,
        )


def test_exact_pdf_page_limit(sample_pdf):
    """Accept PDFs exactly at the maximum page count."""

    result = extract_pdf(
        sample_pdf,
        uuid4(),
        "sample.pdf",
        max_pages=3,
    )

    assert result.page_count == 3


def test_pdf_character_limit(sample_pdf):
    """Reject documents exceeding the extracted character limit."""

    with pytest.raises(ExtractionError, match="character limit"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            "sample.pdf",
            max_characters=10,
        )


def test_exact_pdf_character_limit(sample_pdf):
    """Accept text exactly at the configured character limit."""

    document_id = uuid4()

    original = extract_pdf(
        sample_pdf,
        document_id,
        "sample.pdf",
    )

    limited = extract_pdf(
        sample_pdf,
        document_id,
        "sample.pdf",
        max_characters=original.character_count,
    )

    assert original == limited


def test_pdf_file_size_limit(sample_pdf):
    """Apply the source file size limit during extraction."""

    with pytest.raises(ExtractionError, match="file size limit"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            "sample.pdf",
            max_file_bytes=10,
        )


@pytest.mark.parametrize(
    "limit",
    [0, -1, True, 1.5],
)
def test_invalid_pdf_page_limit(sample_pdf, limit):
    """Reject invalid maximum page arguments."""

    with pytest.raises(ValueError, match="positive integer"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            "sample.pdf",
            max_pages=limit,
        )


@pytest.mark.parametrize(
    "limit",
    [0, -1, True, 1.5],
)
def test_invalid_pdf_character_limit(sample_pdf, limit):
    """Reject invalid maximum character arguments."""

    with pytest.raises(ValueError, match="positive integer"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            "sample.pdf",
            max_characters=limit,
        )


# ==========================================
# Metadata and parser error handling
# ==========================================


@pytest.mark.parametrize(
    "document_id",
    [None, "invalid-uuid", 123],
)
def test_invalid_pdf_document_id(sample_pdf, document_id):
    """Reject invalid document identifiers."""

    with pytest.raises(TypeError, match="must be a UUID"):
        extract_pdf(
            sample_pdf,
            document_id,
            "sample.pdf",
        )


@pytest.mark.parametrize("filename", ["", " ", "\n"])
def test_empty_pdf_filename(sample_pdf, filename):
    """Reject empty document display filenames."""

    with pytest.raises(ValueError, match="cannot be empty"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            filename,
        )


@pytest.mark.parametrize("filename", [None, 123])
def test_invalid_pdf_filename_type(sample_pdf, filename):
    """Reject non-string document display filenames."""

    with pytest.raises(TypeError, match="must be a string"):
        extract_pdf(
            sample_pdf,
            uuid4(),
            filename,
        )


def test_pdf_reader_failure(sample_pdf):
    """Convert unexpected parser failures into ExtractionError."""

    with patch(
        "app.ingestion.extraction.PdfReader",
        side_effect=RuntimeError("Simulated parser failure"),
    ):
        with pytest.raises(
            ExtractionError,
            match="Unable to read",
        ):
            extract_pdf(
                sample_pdf,
                uuid4(),
                "sample.pdf",
            )


def test_pdf_page_extraction_failure(sample_pdf):
    """Report the affected page when text extraction fails."""

    with patch(
        "pypdf.PageObject.extract_text",
        side_effect=RuntimeError("Simulated page failure"),
    ):
        with pytest.raises(
            ExtractionError,
            match="PDF page 1",
        ):
            extract_pdf(
                sample_pdf,
                uuid4(),
                "sample.pdf",
            )
