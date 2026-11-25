"""Tests for the unified document extraction dispatcher."""

from uuid import uuid4

import pytest

from app.ingestion import dispatcher
from app.ingestion.extraction_models import ExtractionError


@pytest.mark.parametrize(
    ("filename", "expected_extractor"),
    [
        ("sample.pdf", "extract_pdf"),
        ("sample.docx", "extract_docx"),
        ("sample.txt", "extract_txt"),
        ("SAMPLE.PDF", "extract_pdf"),
        ("SAMPLE.DOCX", "extract_docx"),
        ("SAMPLE.TXT", "extract_txt"),
    ],
)
def test_correct_extractor_selected(
    tmp_path,
    monkeypatch,
    filename,
    expected_extractor,
):
    path = tmp_path / filename
    path.write_bytes(b"placeholder")

    expected_result = object()
    called = []

    def fake_extractor(*args, **kwargs):
        called.append(True)
        return expected_result

    monkeypatch.setattr(
        dispatcher,
        expected_extractor,
        fake_extractor,
    )

    result = dispatcher.extract_document(
        path,
        uuid4(),
        filename,
    )

    assert result is expected_result
    assert called == [True]


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "sample.exe"
    path.write_bytes(b"test")

    with pytest.raises(ExtractionError):
        dispatcher.extract_document(
            path,
            uuid4(),
            "sample.exe",
        )


def test_mime_mismatch_rejected(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Test.", encoding="utf-8")

    with pytest.raises(ExtractionError):
        dispatcher.extract_document(
            path,
            uuid4(),
            "sample.txt",
            mime_type="application/pdf",
        )


def test_matching_mime_accepted(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Dispatcher test.", encoding="utf-8")

    result = dispatcher.extract_document(
        path,
        uuid4(),
        "sample.txt",
        mime_type="text/plain",
    )

    assert result.has_text
    assert result.mime_type == "text/plain"


def test_invalid_document_id_rejected(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Test.", encoding="utf-8")

    with pytest.raises(TypeError):
        dispatcher.extract_document(
            path,
            "invalid-id",
            "sample.txt",
        )


def test_empty_filename_rejected(tmp_path):
    with pytest.raises(ValueError):
        dispatcher.extract_document(
            tmp_path / "sample.txt",
            uuid4(),
            "",
        )
