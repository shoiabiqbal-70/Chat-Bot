"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for the Production RAG Service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Production RAG Service"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = True

    # Neon PostgreSQL
    neon_database_url: SecretStr | None = None
    neon_migration_database_url: SecretStr | None = None

    # Pinecone
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "production-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Local embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"

    # Gemini
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"

    # RAG settings
    max_upload_size_mb: int = 20
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()


settings = get_settings()