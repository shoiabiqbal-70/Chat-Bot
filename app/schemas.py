"""Pydantic schemas used by the API."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import ConfigDict

class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str = Field(description="Current application status")
    app_name: str = Field(description="Application name")
    environment: str = Field(description="Application environment")


class MessageResponse(BaseModel):
    """Generic API message response."""

    message: str


class DocumentUploadResponse(BaseModel):
    """Represent a successfully registered document."""

    # Read ORM objects directly without exposing local file paths or owner IDs.
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(
        description="Unique document identifier."
    )

    filename: str = Field(
        description="Original sanitized filename."
    )

    sha256: str = Field(
        description="SHA-256 content checksum."
    )

    mime_type: str = Field(
        description="Validated document MIME type."
    )

    status: Literal[
        "pending",
        "processing",
        "completed",
        "failed",
        "deleting",
    ] = Field(
        description="Current document processing status."
    )

    created_at: datetime = Field(
        description="Document registration timestamp."
    )


class DocumentIngestResponse(DocumentUploadResponse):
    """Represent a document that is searchable after ingestion."""

    chunk_count: int = Field(ge=1, description="Number of indexed chunks.")
