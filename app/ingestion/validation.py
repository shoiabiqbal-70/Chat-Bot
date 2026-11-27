"""Validate uploaded documents before storage and processing."""

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from fastapi import UploadFile

from app.config import settings


READ_CHUNK_SIZE = 1024 * 1024

MAX_ARCHIVE_ENTRIES = 5000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    ),
    ".txt": "text/plain",
}


class UploadValidationError(ValueError):
    """Represent a rejected document upload."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    """Contain validated file content and metadata."""

    filename: str
    extension: str
    mime_type: str
    sha256: str
    size_bytes: int
    content: bytes


def sanitize_filename(filename: str | None) -> str:
    """Return a safe display filename without directory components."""
    if not filename:
        raise UploadValidationError("A filename is required.")

    filename = filename.replace("\\", "/")
    filename = PurePosixPath(filename).name

    filename = re.sub(
        r'[\x00-\x1f\x7f<>:"/\\|?*]',
        "_",
        filename,
    ).strip(" .")

    if not filename or filename in {".", ".."}:
        raise UploadValidationError("Invalid filename.")

    if len(filename) > 255:
        raise UploadValidationError("Filename is too long.")

    return filename


def validate_pdf(content: bytes) -> None:
    """Check the basic PDF file signature."""
    if not content.startswith(b"%PDF-"):
        raise UploadValidationError(
            "The uploaded file is not a valid PDF.",
            status_code=415,
        )


def validate_docx(content: bytes) -> None:
    """Check DOCX archive structure and expansion limits."""
    try:
        # DOCX is a ZIP file; cap expanded size and entries before parsing XML.
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()

            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise UploadValidationError(
                    "DOCX archive contains too many entries."
                )

            total_size = sum(
                entry.file_size for entry in entries
            )

            if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise UploadValidationError(
                    "DOCX archive exceeds the expansion limit.",
                    status_code=413,
                )

            names = archive.namelist()

            if len(names) != len(set(names)):
                raise UploadValidationError(
                    "DOCX archive contains duplicate entries."
                )

            for entry in entries:
                # Reject paths and encrypted members that extraction cannot
                # safely or reliably process.
                name = entry.filename.replace("\\", "/")
                path = PurePosixPath(name)

                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or ":" in name
                ):
                    raise UploadValidationError(
                        "Unsafe DOCX archive entry."
                    )

                if entry.flag_bits & 0x1:
                    raise UploadValidationError(
                        "Encrypted DOCX files are not supported."
                    )

            required_entries = {
                "[Content_Types].xml",
                "_rels/.rels",
                "word/document.xml",
            }

            if not required_entries.issubset(set(names)):
                raise UploadValidationError(
                    "Invalid DOCX document structure.",
                    status_code=415,
                )

    except zipfile.BadZipFile:
        raise UploadValidationError(
            "The uploaded file is not a valid DOCX.",
            status_code=415,
        ) from None


def validate_text(content: bytes) -> None:
    """Verify that a text file contains readable UTF-8 text."""
    if b"\x00" in content:
        raise UploadValidationError(
            "Binary content is not allowed in TXT files.",
            status_code=415,
        )

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "TXT files must use UTF-8 encoding.",
            status_code=415,
        ) from None

    if not decoded.strip():
        raise UploadValidationError(
            "The text document is empty."
        )


async def validate_upload(
    upload: UploadFile,
    max_size_bytes: int | None = None,
) -> ValidatedUpload:
    """Read, validate and hash an uploaded document."""
    filename = sanitize_filename(upload.filename)
    extension = PurePosixPath(filename).suffix.lower()

    if extension not in ALLOWED_MIME_TYPES:
        raise UploadValidationError(
            "Only PDF, DOCX and TXT files are supported.",
            status_code=415,
        )

    expected_mime = ALLOWED_MIME_TYPES[extension]

    supplied_mime = (
        (upload.content_type or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    if supplied_mime not in {
        # Browsers may omit a MIME type or use octet-stream for uploads.
        "",
        "application/octet-stream",
        expected_mime,
    }:
        raise UploadValidationError(
            "File extension and MIME type do not match.",
            status_code=415,
        )

    if max_size_bytes is None:
        max_size_bytes = (
            settings.max_upload_size_mb * 1024 * 1024
        )

    if max_size_bytes <= 0:
        raise ValueError(
            "Maximum upload size must be positive."
        )

    if (
        upload.size is not None
        and upload.size > max_size_bytes
    ):
        raise UploadValidationError(
            "File exceeds the maximum upload size.",
            status_code=413,
        )

    await upload.seek(0)

    digest = hashlib.sha256()
    buffer = bytearray()

    while True:
        # Enforce the real byte limit while reading; a client-supplied size
        # header alone cannot be trusted.
        block = await upload.read(READ_CHUNK_SIZE)

        if not block:
            break

        if len(buffer) + len(block) > max_size_bytes:
            raise UploadValidationError(
                "File exceeds the maximum upload size.",
                status_code=413,
            )

        digest.update(block)
        buffer.extend(block)

    if not buffer:
        raise UploadValidationError(
            "Empty files are not allowed."
        )

    content = bytes(buffer)

    if extension == ".pdf":
        validate_pdf(content)
    elif extension == ".docx":
        validate_docx(content)
    elif extension == ".txt":
        validate_text(content)

    await upload.seek(0)

    return ValidatedUpload(
        filename=filename,
        extension=extension,
        mime_type=expected_mime,
        sha256=digest.hexdigest(),
        size_bytes=len(content),
        content=content,
    )
