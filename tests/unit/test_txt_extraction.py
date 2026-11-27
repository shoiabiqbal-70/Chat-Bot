"""Tests for UTF-8 TXT extraction."""

import pytest

from uuid import uuid4

from app.ingestion.extraction_models import (
    BlockType,
    ExtractionError,
)
from app.ingestion.txt_extraction import extract_txt


def test_extract_multiple_paragraphs(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    result = extract_txt(path, uuid4(), "sample.txt")

    assert result.block_count == 2
    assert result.blocks[0].block_index == 0
    assert result.blocks[1].block_index == 1
    assert result.blocks[0].block_type == BlockType.PARAGRAPH
    assert result.full_text == "First paragraph.\n\nSecond paragraph."


def test_unicode_preservation(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text("English\n\nاردو\n\n中文\n\n日本語", encoding="utf-8")

    result = extract_txt(path, uuid4(), "unicode.txt")

    assert "اردو" in result.full_text
    assert "中文" in result.full_text
    assert "日本語" in result.full_text


def test_utf8_bom_support(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfHello world.")

    result = extract_txt(path, uuid4(), "bom.txt")

    assert result.blocks[0].text == "Hello world."


def test_windows_line_endings(tmp_path):
    path = tmp_path / "windows.txt"
    path.write_bytes(b"First\r\n\r\nSecond")

    result = extract_txt(path, uuid4(), "windows.txt")

    assert result.block_count == 2


def test_single_line_break_preserved(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("Line one\nLine two", encoding="utf-8")

    result = extract_txt(path, uuid4(), "lines.txt")

    assert result.block_count == 1
    assert result.blocks[0].text == "Line one\nLine two"


def test_invalid_utf8_rejected(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ExtractionError):
        extract_txt(path, uuid4(), "invalid.txt")


def test_binary_content_rejected(tmp_path):
    path = tmp_path / "binary.txt"
    path.write_bytes(b"Hello\x00World")

    with pytest.raises(ExtractionError):
        extract_txt(path, uuid4(), "binary.txt")


def test_empty_content_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"   \n")

    with pytest.raises(ExtractionError):
        extract_txt(path, uuid4(), "empty.txt")


def test_character_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("A" * 100, encoding="utf-8")

    with pytest.raises(ExtractionError):
        extract_txt(path, uuid4(), "large.txt", max_characters=50)


def test_block_limit(tmp_path):
    path = tmp_path / "blocks.txt"
    path.write_text("First\n\nSecond", encoding="utf-8")

    with pytest.raises(ExtractionError):
        extract_txt(path, uuid4(), "blocks.txt", max_blocks=1)


def test_file_size_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("A" * 100, encoding="utf-8")

    with pytest.raises(ExtractionError):
        extract_txt(
            path,
            uuid4(),
            "large.txt",
            max_file_bytes=50,
        )


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ExtractionError):
        extract_txt(
            tmp_path / "missing.txt",
            uuid4(),
            "missing.txt",
        )


def test_document_metadata(tmp_path):
    path = tmp_path / "metadata.txt"
    path.write_text("Metadata test.", encoding="utf-8")

    document_id = uuid4()

    result = extract_txt(path, document_id, "metadata.txt")

    assert result.document_id == document_id
    assert result.filename == "metadata.txt"
    assert result.mime_type == "text/plain"
    assert result.page_count is None
    assert result.has_text


def test_invalid_document_id(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Test.", encoding="utf-8")

    with pytest.raises(TypeError):
        extract_txt(path, "invalid-id", "sample.txt")
