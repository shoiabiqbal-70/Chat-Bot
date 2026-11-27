
"""Generate normalized embeddings using the configured Sentence Transformer."""

from functools import lru_cache
from typing import Sequence

import numpy as np

from app.config import settings


QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


@lru_cache(maxsize=2)
def _load_model(
    model_name: str,
    device: str,
):
    """Load and cache an embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            model_name,
            device=device,
        )
    except Exception as exc:
        raise EmbeddingError(
            "Unable to load the embedding model."
        ) from exc


@lru_cache(maxsize=2)
def _load_fastembed_model(model_name: str):
    """Load the quantized ONNX model for memory-limited hosting."""
    try:
        from fastembed import TextEmbedding
        # One CPU thread and no CPU memory arena fit Render's free instance.
        return TextEmbedding(model_name=model_name, threads=1,
                             providers=["CPUExecutionProvider"],
                             enable_cpu_mem_arena=False, max_batch_tokens=512)
    except Exception as exc:
        raise EmbeddingError("Unable to load the ONNX embedding model.") from exc


class EmbeddingService:
    """Generate document and query embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        device: str | None = None,
        model=None,
        backend: str | None = None,
    ) -> None:
        """Select a model backend and validate its expected vector dimension."""

        self.model_name = model_name or settings.embedding_model
        self.dimension = (
            dimension
            if dimension is not None
            else settings.embedding_dimension
        )
        self.device = device or settings.embedding_device
        self.backend = backend or settings.embedding_backend

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Invalid embedding model name.")

        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Invalid embedding device.")

        if self.backend not in {"sentence_transformers", "fastembed"}:
            raise ValueError("Invalid embedding backend.")

        self._model = model

    def _get_model(self):
        """Return the cached model and verify its dimensions."""

        if self._model is None:
            self._model = (_load_fastembed_model(self.model_name)
                           if self.backend == "fastembed"
                           else _load_model(self.model_name, self.device))

        actual_dimension = (self._model.embedding_size if self.backend == "fastembed"
                            else self._model.get_sentence_embedding_dimension())

        if actual_dimension != self.dimension:
            # Pinecone index dimensions must match both document and query vectors.
            raise EmbeddingError(
                f"Expected {self.dimension} dimensions, "
                f"received {actual_dimension}."
            )

        return self._model

    def _encode(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        """Encode text and return normalized float32 vectors."""

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if not isinstance(texts, (list, tuple)):
            raise TypeError("Texts must be a list or tuple.")

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Every input must be a string.")

            if not text.strip():
                raise ValueError("Empty text cannot be embedded.")

        model = self._get_model()

        try:
            if self.backend == "fastembed":
                vectors = np.stack(list(model.embed(list(texts), batch_size=min(batch_size, 8))))
            else:
                vectors = model.encode(
                    list(texts),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

            vectors = np.asarray(
                vectors,
                dtype=np.float32,
            )

        except Exception as exc:
            raise EmbeddingError(
                "Embedding generation failed."
            ) from exc

        expected_shape = (
            len(texts),
            self.dimension,
        )

        if vectors.shape != expected_shape:
            raise EmbeddingError(
                f"Invalid embedding shape: {vectors.shape}. "
                f"Expected {expected_shape}."
            )

        if not np.isfinite(vectors).all():
            raise EmbeddingError(
                "Embedding contains invalid numerical values."
            )

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms <= 0):
            raise EmbeddingError(
                "Embedding contains a zero-length vector."
            )

        # Ensure unit-length vectors.
        # Both backends then yield comparable cosine similarity scores.
        vectors = vectors / norms[:, np.newaxis]

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate embeddings for document chunks."""

        return self._encode(
            texts,
            batch_size,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Generate an instruction-prefixed query embedding."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        vectors = self._encode(
            # This instruction is part of the query-side embedding recipe.
            [QUERY_INSTRUCTION + query.strip()],
            batch_size=1,
        )

        return vectors[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the shared embedding service instance."""

    return EmbeddingService()
