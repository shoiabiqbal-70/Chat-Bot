"""Pydantic schemas used by the API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str = Field(description="Current application status")
    app_name: str = Field(description="Application name")
    environment: str = Field(description="Application environment")


class MessageResponse(BaseModel):
    """Generic API message response."""

    message: str
