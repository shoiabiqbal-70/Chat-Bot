"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
    app_debug: bool = False

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

    # Groq
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"

    # RAG settings
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=100)
    min_similarity_score: float = Field(default=0.58, ge=0.0, le=1.0)
    document_storage_dir: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()


settings = get_settings()
