"""Manage local storage of validated uploaded documents."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.config import settings

from app.ingestion.validation import (
    ALLOWED_MIME_TYPES,
    ValidatedUpload,
)


class StorageError(RuntimeError):
    """Represent a document storage operation failure."""


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """Describe a successfully stored document."""

    document_id: UUID
    path: Path
    size_bytes: int
    sha256: str


class LocalDocumentStorage:
    """Store documents using controlled UUID-based filenames."""

    def __init__(
        self,
        root: str | Path | None = None,
    ) -> None:
        """Initialize the configured storage directory."""
        location = Path(root or settings.document_storage_dir).expanduser()
        if not location.is_absolute():
            location = Path(__file__).resolve().parents[2] / location
        self.root = location.resolve()

        try:
            self.root.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
        except OSError as exc:
            raise StorageError(
                "Unable to initialize document storage."
            ) from exc

    def get_path(
        self,
        document_id: UUID,
        extension: str,
    ) -> Path:
        """Construct a controlled document storage path."""
        if not isinstance(document_id, UUID):
            raise ValueError(
                "Document ID must be a UUID."
            )

        extension = extension.lower()

        if extension not in ALLOWED_MIME_TYPES:
            raise ValueError(
                "Unsupported document extension."
            )

        return self.root / (
            f"{document_id.hex}{extension}"
        )

    def save(
        self,
        document_id: UUID,
        upload: ValidatedUpload,
    ) -> StoredDocument:
        """Persist validated content without overwriting files."""
        path = self.get_path(
            document_id,
            upload.extension,
        )

        actual_hash = hashlib.sha256(
            upload.content
        ).hexdigest()

        if actual_hash != upload.sha256:
            raise StorageError(
                "Document checksum verification failed."
            )

        if (
            len(upload.content) != upload.size_bytes
            or not upload.content
        ):
            raise StorageError(
                "Document size verification failed."
            )

        created = False

        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )

            descriptor = os.open(
                path,
                flags,
                0o600,
            )

            created = True

            with os.fdopen(descriptor, "wb") as file:
                file.write(upload.content)
                file.flush()
                os.fsync(file.fileno())

        except FileExistsError as exc:
            raise StorageError(
                "Document storage file already exists."
            ) from exc

        except OSError as exc:
            if created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    raise StorageError(
                        "Storage failed and partial file "
                        "cleanup was unsuccessful."
                    ) from exc

            raise StorageError(
                "Unable to store document."
            ) from exc

        return StoredDocument(
            document_id=document_id,
            path=path,
            size_bytes=upload.size_bytes,
            sha256=upload.sha256,
        )

    def exists(
        self,
        document_id: UUID,
        extension: str,
    ) -> bool:
        """Check whether a stored document exists."""
        path = self.get_path(
            document_id,
            extension,
        )

        return (
            not path.is_symlink()
            and path.is_file()
        )

    def delete(
        self,
        document_id: UUID,
        extension: str,
    ) -> bool:
        """Remove a stored document if it exists."""
        path = self.get_path(
            document_id,
            extension,
        )

        try:
            path.unlink()
            return True

        except FileNotFoundError:
            return False

        except OSError as exc:
            raise StorageError(
                "Unable to delete stored document."
            ) from exc
