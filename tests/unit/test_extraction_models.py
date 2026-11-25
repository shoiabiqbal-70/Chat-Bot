
"""Unit tests for structured document extraction models."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from app.ingestion.extraction_models import (
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
)


PDF_MIME = "application/pdf"

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document"
)

TXT_MIME = "text/plain"


def make_block(**overrides):
    """Create an extracted block with optional field overrides."""

    values = {
        "text": "Sample extracted text.",
        "block_index": 0,
    }

    values.update(overrides)

    return ExtractedBlock(**values)


def make_document(**overrides):
    """Create an extracted document with optional field overrides."""

    values = {
        "document_id": uuid4(),
        "filename": "sample.pdf",
        "mime_type": PDF_MIME,
        "blocks": (make_block(page_number=1),),
        "page_count": 1,
    }

    values.update(overrides)

    return ExtractedDocument(**values)


# ==========================================
# Extracted block tests
# ==========================================


def test_valid_extracted_block():
    """Verify valid block content and metadata."""

    block = ExtractedBlock(
        text="Introduction",
        block_index=0,
        page_number=1,
        section="Chapter One",
        block_type=BlockType.HEADING,
    )

    assert block.text == "Introduction"
    assert block.block_index == 0
    assert block.page_number == 1
    assert block.section == "Chapter One"
    assert block.block_type is BlockType.HEADING


def test_block_default_values():
    """Verify optional block metadata defaults."""

    block = make_block()

    assert block.page_number is None
    assert block.section is None
    assert block.block_type is BlockType.TEXT


def test_block_character_count():
    """Count characters without modifying the original text."""

    block = make_block(text="  Hello world.  ")

    assert block.character_count == 16
    assert block.text == "  Hello world.  "


@pytest.mark.parametrize("block_type", list(BlockType))
def test_supported_block_types(block_type):
    """Accept every defined semantic block type."""

    block = make_block(block_type=block_type)

    assert block.block_type is block_type


@pytest.mark.parametrize(
    "value",
    ["", " ", "\n\t", "   \r\n"],
)
def test_reject_empty_block_text(value):
    """Reject empty or whitespace-only extracted text."""

    with pytest.raises(ValueError, match="cannot be empty"):
        make_block(text=value)


@pytest.mark.parametrize(
    "value",
    [None, 123, b"text"],
)
def test_reject_invalid_block_text_type(value):
    """Reject non-string extracted text."""

    with pytest.raises(TypeError, match="must be a string"):
        make_block(text=value)


def test_reject_negative_block_index():
    """Reject negative block indexes."""

    with pytest.raises(ValueError, match="cannot be negative"):
        make_block(block_index=-1)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "0", None],
)
def test_reject_invalid_block_index_type(value):
    """Reject non-integer block indexes."""

    with pytest.raises(TypeError, match="must be an integer"):
        make_block(block_index=value)


@pytest.mark.parametrize("value", [0, -1])
def test_reject_invalid_page_number(value):
    """Reject zero and negative page numbers."""

    with pytest.raises(ValueError, match="at least one"):
        make_block(page_number=value)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "1"],
)
def test_reject_invalid_page_number_type(value):
    """Reject non-integer page numbers."""

    with pytest.raises(TypeError, match="must be an integer"):
        make_block(page_number=value)


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_reject_empty_section(value):
    """Reject empty section headings."""

    with pytest.raises(ValueError, match="cannot be empty"):
        make_block(section=value)


@pytest.mark.parametrize("value", [123, [], True])
def test_reject_invalid_section_type(value):
    """Reject non-string section metadata."""

    with pytest.raises(TypeError, match="must be a string"):
        make_block(section=value)


@pytest.mark.parametrize(
    "value",
    ["heading", "unknown", None],
)
def test_reject_invalid_block_type(value):
    """Require an actual BlockType enumeration value."""

    with pytest.raises(TypeError, match="BlockType"):
        make_block(block_type=value)


def test_block_immutability():
    """Prevent modification of extracted block fields."""

    block = make_block()

    with pytest.raises(FrozenInstanceError):
        block.text = "Modified"


# ==========================================
# Extracted document tests
# ==========================================


def test_valid_extracted_document():
    """Verify document identity and extraction metadata."""

    document_id = uuid4()

    document = make_document(document_id=document_id)

    assert document.document_id == document_id
    assert document.filename == "sample.pdf"
    assert document.mime_type == PDF_MIME
    assert document.page_count == 1
    assert document.block_count == 1
    assert document.has_text


@pytest.mark.parametrize(
    ("filename", "mime_type", "page_count", "page_number"),
    [
        ("sample.pdf", PDF_MIME, 1, 1),
        ("sample.docx", DOCX_MIME, None, None),
        ("sample.txt", TXT_MIME, None, None),
    ],
)
def test_supported_document_formats(
    filename,
    mime_type,
    page_count,
    page_number,
):
    """Accept every supported document format."""

    document = make_document(
        filename=filename,
        mime_type=mime_type,
        page_count=page_count,
        blocks=(make_block(page_number=page_number),),
    )

    assert document.mime_type == mime_type
    assert document.page_count == page_count


@pytest.mark.parametrize(
    "value",
    [None, "invalid-uuid", 123],
)
def test_reject_invalid_document_id(value):
    """Require a UUID document identifier."""

    with pytest.raises(TypeError, match="must be a UUID"):
        make_document(document_id=value)


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_reject_empty_filename(value):
    """Reject missing or whitespace-only filenames."""

    with pytest.raises(ValueError, match="cannot be empty"):
        make_document(filename=value)


@pytest.mark.parametrize("value", [None, 123])
def test_reject_invalid_filename_type(value):
    """Reject non-string filenames."""

    with pytest.raises(TypeError, match="must be a string"):
        make_document(filename=value)


@pytest.mark.parametrize(
    "value",
    ["application/json", "image/png", ""],
)
def test_reject_unsupported_mime_type(value):
    """Reject unsupported document MIME types."""

    with pytest.raises(ValueError, match="Unsupported"):
        make_document(mime_type=value)


def test_reject_mutable_block_collection():
    """Require an immutable tuple of extracted blocks."""

    with pytest.raises(TypeError, match="must be a tuple"):
        make_document(blocks=[make_block()])


def test_reject_invalid_block_object():
    """Reject objects that are not ExtractedBlock instances."""

    with pytest.raises(TypeError, match="Invalid extracted block"):
        make_document(blocks=("invalid block",))


def test_reject_mutable_warning_collection():
    """Require extraction warnings to be stored in a tuple."""

    with pytest.raises(TypeError, match="must be a tuple"):
        make_document(warnings=["Warning"])


@pytest.mark.parametrize("value", [None, 123, True])
def test_reject_invalid_warning_type(value):
    """Require every extraction warning to be a string."""

    with pytest.raises(TypeError, match="must be strings"):
        make_document(warnings=(value,))


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_reject_empty_warning(value):
    """Reject empty extraction warning messages."""

    with pytest.raises(ValueError, match="cannot be empty"):
        make_document(warnings=(value,))


def test_reject_negative_page_count():
    """Reject negative document page counts."""

    with pytest.raises(ValueError, match="cannot be negative"):
        make_document(page_count=-1)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "1"],
)
def test_reject_invalid_page_count_type(value):
    """Reject non-integer page counts."""

    with pytest.raises(TypeError, match="must be an integer"):
        make_document(page_count=value)


# ==========================================
# Ordering and consistency tests
# ==========================================


def test_sequential_block_order():
    """Preserve blocks in their original document order."""

    blocks = (
        make_block(text="First", block_index=0, page_number=1),
        make_block(text="Second", block_index=1, page_number=1),
        make_block(text="Third", block_index=2, page_number=2),
    )

    document = make_document(
        blocks=blocks,
        page_count=2,
    )

    assert document.blocks == blocks
    assert document.block_count == 3


@pytest.mark.parametrize(
    "indexes",
    [
        (1,),
        (0, 0),
        (0, 2),
        (1, 0),
    ],
)
def test_reject_nonsequential_block_indexes(indexes):
    """Reject duplicate, missing or out-of-order block indexes."""

    blocks = tuple(
        make_block(block_index=index)
        for index in indexes
    )

    with pytest.raises(ValueError, match="must be sequential"):
        make_document(blocks=blocks)


def test_reject_page_exceeding_document_count():
    """Reject block pages outside the document page range."""

    with pytest.raises(ValueError, match="exceeds"):
        make_document(
            blocks=(make_block(page_number=3),),
            page_count=2,
        )


def test_reject_reverse_page_order():
    """Reject decreasing page numbers."""

    blocks = (
        make_block(block_index=0, page_number=2),
        make_block(block_index=1, page_number=1),
    )

    with pytest.raises(ValueError, match="document order"):
        make_document(
            blocks=blocks,
            page_count=2,
        )


def test_allow_skipped_page_numbers():
    """Allow blank pages between extracted text blocks."""

    blocks = (
        make_block(block_index=0, page_number=1),
        make_block(block_index=1, page_number=4),
    )

    document = make_document(
        blocks=blocks,
        page_count=4,
    )

    assert document.block_count == 2
    assert document.blocks[1].page_number == 4


def test_document_without_page_metadata():
    """Represent DOCX sections without fabricated page numbers."""

    document = make_document(
        filename="sample.docx",
        mime_type=DOCX_MIME,
        page_count=None,
        blocks=(
            make_block(
                section="Introduction",
                page_number=None,
            ),
        ),
    )

    assert document.page_count is None
    assert document.blocks[0].page_number is None
    assert document.blocks[0].section == "Introduction"


# ==========================================
# Document properties and warning tests
# ==========================================


def test_document_character_count():
    """Count extracted characters across all blocks."""

    blocks = (
        make_block(text="Hello", block_index=0),
        make_block(text="World", block_index=1),
    )

    document = make_document(blocks=blocks)

    assert document.character_count == 10


def test_full_text_reconstruction():
    """Combine blocks with deterministic paragraph separators."""

    blocks = (
        make_block(text="First paragraph.", block_index=0),
        make_block(text="Second paragraph.", block_index=1),
    )

    document = make_document(blocks=blocks)

    assert document.full_text == (
        "First paragraph.\n\nSecond paragraph."
    )


def test_empty_document_representation():
    """Represent a document with no extractable text."""

    document = make_document(
        blocks=(),
        page_count=2,
        warnings=("No extractable text found.",),
    )

    assert document.block_count == 0
    assert document.character_count == 0
    assert document.full_text == ""
    assert document.has_text is False
    assert document.page_count == 2


def test_warning_preservation():
    """Preserve extraction warnings in their original order."""

    warnings = (
        "Page 2 contains no extractable text.",
        "Page 4 contains no extractable text.",
    )

    document = make_document(warnings=warnings)

    assert document.warnings == warnings


def test_default_document_warnings():
    """Use an empty tuple when no warnings are present."""

    document = make_document()

    assert document.warnings == ()


def test_document_immutability():
    """Prevent modification of extracted document fields."""

    document = make_document()

    with pytest.raises(FrozenInstanceError):
        document.filename = "modified.pdf"


def test_extraction_error_definition():
    """Verify the extraction exception hierarchy."""

    assert issubclass(ExtractionError, RuntimeError)

    with pytest.raises(ExtractionError, match="Extraction failed"):
        raise ExtractionError("Extraction failed")
