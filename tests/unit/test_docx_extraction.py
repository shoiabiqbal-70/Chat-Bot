"""Unit tests for DOCX document extraction."""

from uuid import uuid4

import pytest
from docx import Document

from app.ingestion.docx_extraction import extract_docx
from app.ingestion.extraction_models import (
    BlockType,
    ExtractionError,
)


@pytest.fixture
def docx_file(tmp_path):
    """Create a representative DOCX document."""

    path = tmp_path / "sample.docx"

    document = Document()

    document.add_heading("Introduction", level=1)

    document.add_paragraph(
        "This is the introduction paragraph."
    )

    document.add_heading("Results", level=2)

    document.add_paragraph(
        "These are the research results."
    )

    table = document.add_table(rows=2, cols=2)

    table.cell(0, 0).text = "Parameter"
    table.cell(0, 1).text = "Value"

    table.cell(1, 0).text = "Accuracy"
    table.cell(1, 1).text = "95%"

    document.add_paragraph(
        "Final document conclusion."
    )

    document.save(path)

    return path


def test_extract_docx_metadata(docx_file):
    """Verify document metadata."""

    document_id = uuid4()

    result = extract_docx(
        docx_file,
        document_id,
        "sample.docx",
    )

    assert result.document_id == document_id
    assert result.filename == "sample.docx"
    assert result.mime_type == (
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    )

    assert result.page_count is None
    assert result.has_text


def test_extract_headings(docx_file):
    """Verify heading recognition."""

    result = extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    headings = [
        block.text
        for block in result.blocks
        if block.block_type == BlockType.HEADING
    ]

    assert headings == [
        "Introduction",
        "Results",
    ]


def test_extract_paragraphs(docx_file):
    """Verify paragraph extraction."""

    result = extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    paragraphs = [
        block.text
        for block in result.blocks
        if block.block_type == BlockType.PARAGRAPH
    ]

    assert "This is the introduction paragraph." in paragraphs
    assert "These are the research results." in paragraphs
    assert "Final document conclusion." in paragraphs


def test_extract_tables(docx_file):
    """Verify table extraction."""

    result = extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    tables = [
        block
        for block in result.blocks
        if block.block_type == BlockType.TABLE
    ]

    assert len(tables) == 1

    assert "Parameter" in tables[0].text
    assert "Accuracy" in tables[0].text
    assert "95%" in tables[0].text


def test_preserve_document_order(docx_file):
    """Verify paragraph and table ordering."""

    result = extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    texts = [block.text for block in result.blocks]

    assert texts.index("Introduction") < texts.index(
        "This is the introduction paragraph."
    )

    table_index = next(
        index
        for index, block in enumerate(result.blocks)
        if block.block_type == BlockType.TABLE
    )

    conclusion_index = texts.index(
        "Final document conclusion."
    )

    assert table_index < conclusion_index

    assert [
        block.block_index for block in result.blocks
    ] == list(range(len(result.blocks)))


def test_section_metadata(docx_file):
    """Verify headings are associated with subsequent content."""

    result = extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    introduction = next(
        block
        for block in result.blocks
        if block.text == "This is the introduction paragraph."
    )

    conclusion = next(
        block
        for block in result.blocks
        if block.text == "Final document conclusion."
    )

    assert introduction.section == "Introduction"
    assert conclusion.section == "Results"


def test_empty_docx(tmp_path):
    """Handle documents without extractable text."""

    path = tmp_path / "empty.docx"

    Document().save(path)

    result = extract_docx(
        path,
        uuid4(),
        "empty.docx",
    )

    assert not result.has_text
    assert result.block_count == 0
    assert result.character_count == 0
    assert result.full_text == ""
    assert len(result.warnings) == 1


def test_character_limit(docx_file):
    """Reject documents exceeding character limits."""

    with pytest.raises(ExtractionError):
        extract_docx(
            docx_file,
            uuid4(),
            "sample.docx",
            max_characters=10,
        )


def test_block_limit(docx_file):
    """Reject documents exceeding block limits."""

    with pytest.raises(ExtractionError):
        extract_docx(
            docx_file,
            uuid4(),
            "sample.docx",
            max_blocks=1,
        )


def test_invalid_docx(tmp_path):
    """Reject invalid DOCX archives."""

    path = tmp_path / "invalid.docx"
    path.write_bytes(b"This is not a DOCX document.")

    with pytest.raises(ExtractionError):
        extract_docx(
            path,
            uuid4(),
            "invalid.docx",
        )


def test_original_file_unchanged(docx_file):
    """Verify extraction never modifies the source."""

    original = docx_file.read_bytes()

    extract_docx(
        docx_file,
        uuid4(),
        "sample.docx",
    )

    assert docx_file.read_bytes() == original


@pytest.mark.parametrize(
    "limit",
    [0, -1],
)
def test_invalid_character_limit(docx_file, limit):
    """Reject invalid extraction limits."""

    with pytest.raises(ValueError):
        extract_docx(
            docx_file,
            uuid4(),
            "sample.docx",
            max_characters=limit,
        )


