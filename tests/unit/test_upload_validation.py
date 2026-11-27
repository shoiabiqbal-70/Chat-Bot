"""Unit tests for document upload validation."""

import asyncio
import hashlib
import io
import zipfile

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.ingestion import validation
from app.ingestion.validation import (
    UploadValidationError,
    sanitize_filename,
    validate_upload,
)


PDF_MIME = "application/pdf"

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document"
)

TXT_MIME = "text/plain"


def make_docx(
    extra_entries=None,
    missing_entry=None,
) -> bytes:
    """Create a minimal DOCX-like archive for validation tests."""
    buffer = io.BytesIO()

    entries = {
        "[Content_Types].xml": b"<Types/>",
        "_rels/.rels": b"<Relationships/>",
        "word/document.xml": b"<document/>",
    }

    if missing_entry:
        entries.pop(missing_entry, None)

    if extra_entries:
        entries.update(extra_entries)

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)

    return buffer.getvalue()


def run_validation(
    filename,
    content,
    mime_type,
    max_size_bytes=None,
    reported_size=None,
):
    """Run asynchronous validation and close the upload."""

    async def run():
        headers = Headers(
            {"content-type": mime_type}
            if mime_type is not None
            else {}
        )

        upload = UploadFile(
            file=io.BytesIO(content),
            filename=filename,
            headers=headers,
            size=reported_size,
        )

        try:
            return await validate_upload(
                upload,
                max_size_bytes=max_size_bytes,
            )
        finally:
            await upload.close()

    return asyncio.run(run())


# ==========================================
# Valid uploads
# ==========================================


@pytest.mark.parametrize(
    "filename, content, mime_type",
    [
        ("sample.txt", b"Hello RAG", TXT_MIME),
        ("sample.pdf", b"%PDF-1.7\n%%EOF", PDF_MIME),
        ("sample.docx", make_docx(), DOCX_MIME),
    ],
)
def test_valid_upload(filename, content, mime_type):
    """Accept supported files with valid basic signatures."""
    result = run_validation(
        filename,
        content,
        mime_type,
    )

    assert result.filename == filename
    assert result.content == content
    assert result.mime_type == mime_type
    assert result.size_bytes == len(content)

    assert result.sha256 == hashlib.sha256(
        content
    ).hexdigest()


def test_uppercase_extension():
    """Accept uppercase supported extensions."""
    result = run_validation(
        "DOCUMENT.PDF",
        b"%PDF-1.7",
        PDF_MIME,
    )

    assert result.extension == ".pdf"


def test_generic_mime_type():
    """Accept generic MIME when content checks pass."""
    result = run_validation(
        "sample.txt",
        b"Valid text",
        "application/octet-stream",
    )

    assert result.mime_type == TXT_MIME


def test_missing_mime_type():
    """Accept missing MIME when content checks pass."""
    result = run_validation(
        "sample.txt",
        b"Valid text",
        None,
    )

    assert result.mime_type == TXT_MIME


def test_mime_with_charset():
    """Accept MIME parameters for UTF-8 text."""
    result = run_validation(
        "sample.txt",
        b"Valid text",
        "text/plain; charset=utf-8",
    )

    assert result.mime_type == TXT_MIME


def test_utf8_unicode_text():
    """Accept non-ASCII UTF-8 document content."""
    content = "Hello \u4e16\u754c".encode("utf-8")

    result = run_validation(
        "unicode.txt",
        content,
        TXT_MIME,
    )

    assert result.content == content


# ==========================================
# Filename validation
# ==========================================


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("../../sample.pdf", "sample.pdf"),
        (r"..\..\sample.pdf", "sample.pdf"),
        (r"C:\documents\sample.pdf", "sample.pdf"),
        ("sample?.txt", "sample_.txt"),
        ("sample*.txt", "sample_.txt"),
    ],
)
def test_filename_sanitization(filename, expected):
    """Remove directory components and unsafe characters."""
    assert sanitize_filename(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [None, "", ".", "..", "   "],
)
def test_invalid_filename(filename):
    """Reject missing or unusable filenames."""
    with pytest.raises(UploadValidationError):
        sanitize_filename(filename)


def test_filename_too_long():
    """Reject filenames exceeding the database limit."""
    with pytest.raises(UploadValidationError):
        sanitize_filename("a" * 256 + ".pdf")


# ==========================================
# Invalid uploads
# ==========================================


@pytest.mark.parametrize(
    "filename, content, mime_type, expected_status",
    [
        ("program.exe", b"binary", "application/octet-stream", 415),
        ("image.png", b"image", "image/png", 415),
        ("sample.pdf", b"Invalid PDF", PDF_MIME, 415),
        ("sample.docx", b"Invalid ZIP", DOCX_MIME, 415),
        ("sample.txt", b"", TXT_MIME, 400),
        ("sample.txt", b"   \n\t", TXT_MIME, 400),
        ("sample.txt", b"\x00binary", TXT_MIME, 415),
        ("sample.txt", b"\xff\xfe", TXT_MIME, 415),
        ("sample.pdf", b"%PDF-1.7", TXT_MIME, 415),
    ],
)
def test_invalid_upload(
    filename,
    content,
    mime_type,
    expected_status,
):
    """Reject unsupported, malformed or empty uploads."""
    with pytest.raises(UploadValidationError) as error:
        run_validation(
            filename,
            content,
            mime_type,
        )

    assert error.value.status_code == expected_status


# ==========================================
# Upload size restrictions
# ==========================================


def test_exact_size_limit():
    """Accept a file exactly equal to the size limit."""
    result = run_validation(
        "sample.txt",
        b"12345",
        TXT_MIME,
        max_size_bytes=5,
    )

    assert result.size_bytes == 5


def test_exceeds_size_limit():
    """Reject content exceeding the configured limit."""
    with pytest.raises(UploadValidationError) as error:
        run_validation(
            "sample.txt",
            b"123456",
            TXT_MIME,
            max_size_bytes=5,
        )

    assert error.value.status_code == 413


def test_reported_size_exceeds_limit():
    """Reject an oversized reported upload."""
    with pytest.raises(UploadValidationError) as error:
        run_validation(
            "sample.txt",
            b"12345",
            TXT_MIME,
            max_size_bytes=5,
            reported_size=100,
        )

    assert error.value.status_code == 413


def test_invalid_size_configuration():
    """Reject a nonpositive size configuration."""
    with pytest.raises(ValueError):
        run_validation(
            "sample.txt",
            b"Valid text",
            TXT_MIME,
            max_size_bytes=0,
        )


# ==========================================
# DOCX archive security
# ==========================================


def test_docx_missing_required_entry():
    """Reject archives missing required DOCX entries."""
    content = make_docx(
        missing_entry="word/document.xml",
    )

    with pytest.raises(UploadValidationError) as error:
        run_validation(
            "sample.docx",
            content,
            DOCX_MIME,
        )

    assert error.value.status_code == 415


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.xml",
        "/absolute.xml",
        "C:/outside.xml",
        r"..\outside.xml",
    ],
)
def test_docx_unsafe_archive_paths(unsafe_path):
    """Reject unsafe paths inside DOCX archives."""
    content = make_docx(
        extra_entries={
            unsafe_path: b"Unsafe content",
        },
    )

    with pytest.raises(UploadValidationError):
        run_validation(
            "sample.docx",
            content,
            DOCX_MIME,
        )


def test_docx_archive_entry_limit(monkeypatch):
    """Reject archives containing excessive entries."""
    monkeypatch.setattr(
        validation,
        "MAX_ARCHIVE_ENTRIES",
        2,
    )

    with pytest.raises(UploadValidationError):
        run_validation(
            "sample.docx",
            make_docx(),
            DOCX_MIME,
        )


def test_docx_archive_expansion_limit(monkeypatch):
    """Reject excessive declared decompressed content."""
    monkeypatch.setattr(
        validation,
        "MAX_ARCHIVE_UNCOMPRESSED_BYTES",
        10,
    )

    with pytest.raises(UploadValidationError) as error:
        run_validation(
            "sample.docx",
            make_docx(),
            DOCX_MIME,
        )

    assert error.value.status_code == 413


def test_docx_duplicate_entries():
    """Reject archives containing duplicate entry names."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            b"<Types/>",
        )

        archive.writestr(
            "_rels/.rels",
            b"<Relationships/>",
        )

        archive.writestr(
            "word/document.xml",
            b"<document/>",
        )

        with pytest.warns(UserWarning):
            archive.writestr(
                "word/document.xml",
                b"<duplicate/>",
            )

    with pytest.raises(UploadValidationError):
        run_validation(
            "sample.docx",
            buffer.getvalue(),
            DOCX_MIME,
        )


# ==========================================
# Duplicate detection support
# ==========================================


def test_identical_content_has_identical_hash():
    """Generate identical hashes regardless of filename."""
    first = run_validation(
        "first.txt",
        b"Identical content",
        TXT_MIME,
    )

    second = run_validation(
        "second.txt",
        b"Identical content",
        TXT_MIME,
    )

    assert first.sha256 == second.sha256


def test_different_content_has_different_hash():
    """Generate different hashes for different content."""
    first = run_validation(
        "sample.txt",
        b"First content",
        TXT_MIME,
    )

    second = run_validation(
        "sample.txt",
        b"Second content",
        TXT_MIME,
    )

    assert first.sha256 != second.sha256
