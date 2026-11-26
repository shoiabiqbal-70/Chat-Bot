
"""Unit tests for embedding generation."""

import numpy as np
import pytest

from app.ingestion import embedding
from app.ingestion.embedding import (
    EmbeddingError,
    EmbeddingService,
    QUERY_INSTRUCTION,
)


class FakeModel:
    """Mock embedding model for isolated tests."""

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.last_texts = None
        self.last_options = None
        
    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.last_texts = texts
        self.last_options = kwargs

        return np.tile(
            np.array([[3, 4, 0, 0]], dtype=np.float32),
            (len(texts), 1),
        )


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def service(fake_model):
    return EmbeddingService(
        model_name="test-model",
        dimension=4,
        device="cpu",
        model=fake_model,
    )


def test_document_embedding_shape(service):
    vectors = service.embed_documents([
        "First document.",
        "Second document.",
    ])

    assert vectors.shape == (2, 4)


def test_embedding_dimension(service):
    vectors = service.embed_documents([
        "Test document."
    ])

    assert vectors.shape[1] == 4


def test_embedding_dtype(service):
    vectors = service.embed_documents([
        "Test document."
    ])

    assert vectors.dtype == np.float32


def test_embeddings_are_normalized(service):
    vectors = service.embed_documents([
        "First document.",
        "Second document.",
    ])

    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_document_text_is_not_modified(service, fake_model):
    service.embed_documents([
        "Original document text."
    ])

    assert fake_model.last_texts == [
        "Original document text."
    ]


def test_query_instruction_is_applied(service, fake_model):
    service.embed_query("What is artificial intelligence?")

    assert fake_model.last_texts == [
        QUERY_INSTRUCTION + "What is artificial intelligence?"
    ]


def test_query_returns_single_vector(service):
    vector = service.embed_query("Sample query.")

    assert vector.shape == (4,)
    assert vector.dtype == np.float32


def test_empty_document_batch(service):
    vectors = service.embed_documents([])

    assert vectors.shape == (0, 4)


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["   "],
    ],
)
def test_empty_text_rejected(service, texts):
    with pytest.raises(ValueError):
        service.embed_documents(texts)


def test_invalid_text_type(service):
    with pytest.raises(TypeError):
        service.embed_documents([123])


def test_invalid_input_collection(service):
    with pytest.raises(TypeError):
        service.embed_documents("Not a list")


def test_invalid_batch_size(service):
    with pytest.raises(ValueError):
        service.embed_documents(
            ["Sample text."],
            batch_size=0,
        )


def test_invalid_query(service):
    with pytest.raises(ValueError):
        service.embed_query("")


def test_model_dimension_mismatch(fake_model):
    fake_model.dimension = 3

    service = EmbeddingService(
        model_name="test-model",
        dimension=4,
        device="cpu",
        model=fake_model,
    )

    with pytest.raises(EmbeddingError):
        service.embed_documents(["Sample text."])


def test_invalid_embedding_shape(fake_model):
    fake_model.encode = lambda texts, **kwargs: np.zeros(
        (len(texts), 3),
        dtype=np.float32,
    )

    service = EmbeddingService(
        model_name="test-model",
        dimension=4,
        model=fake_model,
    )

    with pytest.raises(EmbeddingError):
        service.embed_documents(["Sample text."])


def test_non_finite_embedding(fake_model):
    fake_model.encode = lambda texts, **kwargs: np.full(
        (len(texts), 4),
        np.nan,
        dtype=np.float32,
    )

    service = EmbeddingService(
        model_name="test-model",
        dimension=4,
        model=fake_model,
    )

    with pytest.raises(EmbeddingError):
        service.embed_documents(["Sample text."])


def test_zero_embedding_rejected(fake_model):
    fake_model.encode = lambda texts, **kwargs: np.zeros(
        (len(texts), 4),
        dtype=np.float32,
    )

    service = EmbeddingService(
        model_name="test-model",
        dimension=4,
        model=fake_model,
    )

    with pytest.raises(EmbeddingError):
        service.embed_documents(["Sample text."])


def test_invalid_dimension():
    with pytest.raises(ValueError):
        EmbeddingService(dimension=0)


def test_model_loading_cached(monkeypatch):
    calls = []

    def fake_loader(model_name, device):
        calls.append((model_name, device))
        return FakeModel()

    monkeypatch.setattr(
        embedding,
        "_load_model",
        fake_loader,
    )

    service = EmbeddingService(
        model_name="test-model",
        dimension=4,
        device="cpu",
    )

    service.embed_documents(["First document."])
    service.embed_documents(["Second document."])

    assert len(calls) == 1


def test_fastembed_backend_preserves_shape_and_query_prefix():
    class FakeFastEmbed:
        embedding_size = 4

        def __init__(self):
            self.last_texts = None

        def embed(self, texts, batch_size):
            self.last_texts = texts
            return iter([np.array([3, 4, 0, 0], dtype=np.float32) for _ in texts])

    model = FakeFastEmbed()
    service = EmbeddingService(model_name="test-model", dimension=4,
                               model=model, backend="fastembed")
    vectors = service.embed_documents(["first", "second"])
    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert service.embed_query("question").shape == (4,)
    assert model.last_texts == [QUERY_INSTRUCTION + "question"]
