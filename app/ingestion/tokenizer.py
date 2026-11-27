"""Load and cache the configured embedding tokenizer."""

from functools import lru_cache

from tokenizers import Tokenizer

from app.config import settings


@lru_cache(maxsize=2)
def get_tokenizer(model_name=None):
    """Load the tokenizer associated with the embedding model."""
    # Chunk sizes are measured with the same token vocabulary as embeddings.

    model_name = model_name or settings.embedding_model

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("A valid embedding model name is required.")

    try:
        return Tokenizer.from_pretrained(model_name)

    except Exception as exc:
        raise RuntimeError(
            "Unable to load the embedding tokenizer. "
            "Check model availability and network connectivity."
        ) from exc
