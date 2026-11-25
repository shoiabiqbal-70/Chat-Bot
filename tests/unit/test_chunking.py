
"""Tests for token-aware document chunking."""

from uuid import uuid4

import pytest

from app.ingestion.chunking import (
    chunk_document,
    DocumentChunk,
)

from app.ingestion.extraction_models import (
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
)



class FakeEncoding:
    """Represent encoded tokens using integer IDs."""

    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Deterministic integer-based tokenizer for unit tests."""

    def __init__(self):
        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self.next_id = 1

    def encode(self, text):
        """Convert whitespace-separated words into integer IDs."""

        ids = []

        for token in text.split():
            if token not in self.vocabulary:
                token_id = self.next_id
                self.vocabulary[token] = token_id
                self.reverse_vocabulary[token_id] = token
                self.next_id += 1

            ids.append(self.vocabulary[token])

        return FakeEncoding(ids)

    def decode(self, ids):
        """Convert integer IDs back into readable text."""

        try:
            return " ".join(
                self.reverse_vocabulary[token_id]
                for token_id in ids
            )
        except KeyError as exc:
            raise ValueError(
                f"Unknown token ID: {exc}"
            ) from exc



@pytest.fixture
def tokenizer():
    """Provide a fake tokenizer."""
    return FakeTokenizer()


def make_document(blocks):
    """Create an extracted document for testing."""

    return ExtractedDocument(
        document_id=uuid4(),
        filename="sample.txt",
        mime_type="text/plain",
        blocks=tuple(blocks),
    )


def make_block(
    text,
    index,
    page=None,
    section=None,
    block_type=BlockType.PARAGRAPH,
):
    """Create an extracted document block."""

    return ExtractedBlock(
        text=text,
        block_index=index,
        page_number=page,
        section=section,
        block_type=block_type,
    )


def test_empty_document(tokenizer):
    """Return no chunks for an empty document."""

    document = make_document([])

    assert chunk_document(document, tokenizer) == []


def test_single_block(tokenizer):
    """Extract a single block correctly."""

    document = make_document([
        make_block("This is a simple test.", 0)
    ])

    chunks = chunk_document(document, tokenizer)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "This is a simple test."
    assert chunks[0].token_count == 5
    assert chunks[0].source_block_indices == (0,)


def test_chunk_preserves_page_metadata(tokenizer):
    """Preserve original PDF page numbers."""

    document = make_document([
        make_block("First page content.", 0, page=1),
        make_block("Second page content.", 1, page=2),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=20,
        overlap=2,
    )

    assert len(chunks) == 2
    assert [c.page_number for c in chunks] == [1, 2]


def test_chunk_preserves_section_metadata(tokenizer):
    """Preserve DOCX section headings."""

    document = make_document([
        make_block(
            "Introduction content.",
            0,
            section="Introduction",
        ),
        make_block(
            "Results content.",
            1,
            section="Results",
        ),
    ])

    chunks = chunk_document(document, tokenizer)

    assert len(chunks) == 2
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Results"


def test_chunk_indexes_are_sequential(tokenizer):
    """Ensure generated chunk indexes are sequential."""

    document = make_document([
        make_block("First paragraph.", 0),
        make_block("Second paragraph.", 1),
        make_block("Third paragraph.", 2),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=3,
        overlap=0,
    )

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_chunk_size_is_respected(tokenizer):
    """Never exceed the configured token limit."""

    document = make_document([
        make_block(
            "one two three four five six seven eight nine ten.",
            0,
        )
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) > 1

    assert all(
        chunk.token_count <= 4
        for chunk in chunks
    )


def test_overlap_is_preserved(tokenizer):
    """Preserve the configured overlap between chunks."""

    document = make_document([
        make_block(
            "one two three four five six seven eight nine ten.",
            0,
        )
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=4,
        overlap=2,
    )

    assert len(chunks) > 1

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()

    assert first_tokens[-2:] == second_tokens[:2]


def test_overlap_does_not_cross_pages(tokenizer):
    """Never carry overlapping text across PDF pages."""

    document = make_document([
        make_block(
            "one two three four five.",
            0,
            page=1,
        ),
        make_block(
            "six seven eight nine ten.",
            1,
            page=2,
        ),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=3,
        overlap=2,
    )

    page_one = [
        c for c in chunks if c.page_number == 1
    ]

    page_two = [
        c for c in chunks if c.page_number == 2
    ]

    assert page_one
    assert page_two

    assert all(
        "six" not in c.text.split()
        for c in page_one
    )

    assert all(
        "one" not in c.text.split()
        for c in page_two
    )


def test_overlap_does_not_cross_sections(tokenizer):
    """Never carry overlapping text across DOCX sections."""

    document = make_document([
        make_block(
            "one two three four five.",
            0,
            section="A",
        ),
        make_block(
            "six seven eight nine ten.",
            1,
            section="B",
        ),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=3,
        overlap=2,
    )

    section_a = [
        c for c in chunks if c.section == "A"
    ]

    section_b = [
        c for c in chunks if c.section == "B"
    ]

    assert section_a
    assert section_b

    assert all(
        "six" not in c.text.split()
        for c in section_a
    )

    assert all(
        "one" not in c.text.split()
        for c in section_b
    )


def test_long_sentence_is_split(tokenizer):
    """Split unusually long sentences into safe chunks."""

    document = make_document([
        make_block(
            "one two three four five six seven eight nine ten.",
            0,
        )
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) > 1

    assert all(
        chunk.token_count <= 4
        for chunk in chunks
    )


def test_source_block_metadata(tokenizer):
    """Preserve original source block indexes."""

    document = make_document([
        make_block("First sentence.", 0),
        make_block("Second sentence.", 1),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=3,
        overlap=0,
    )

    assert len(chunks) == 2

    assert chunks[0].source_block_indices == (0,)
    assert chunks[1].source_block_indices == (1,)


def test_multiple_blocks_can_share_chunk(tokenizer):
    """Preserve all source indexes when blocks are combined."""

    document = make_document([
        make_block("First paragraph.", 0),
        make_block("Second paragraph.", 1),
    ])

    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=20,
        overlap=0,
    )

    assert len(chunks) == 1

    assert chunks[0].source_block_indices == (0, 1)


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (0, 0),
        (1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_settings(
    tokenizer,
    chunk_size,
    overlap,
):
    """Reject invalid chunking configuration."""

    document = make_document([
        make_block("Sample content.", 0)
    ])

    with pytest.raises(ValueError):
        chunk_document(
            document,
            tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_invalid_document(tokenizer):
    """Reject objects that are not extracted documents."""

    with pytest.raises(TypeError):
        chunk_document("invalid", tokenizer)


def test_invalid_tokenizer():
    """Reject objects without tokenizer methods."""

    document = make_document([
        make_block("Sample content.", 0)
    ])

    with pytest.raises(TypeError):
        chunk_document(document, object())


def test_document_chunk_validation():
    """Reject empty chunk content."""

    with pytest.raises(ValueError):
        DocumentChunk(
            text=" ",
            chunk_index=0,
            page_number=None,
            section=None,
            token_count=1,
            source_block_indices=(0,),
        )


def test_negative_chunk_index():
    """Reject negative chunk indexes."""

    with pytest.raises(ValueError):
        DocumentChunk(
            text="Sample content",
            chunk_index=-1,
            page_number=None,
            section=None,
            token_count=2,
            source_block_indices=(0,),
        )


def test_missing_source_metadata():
    """Reject chunks without source block references."""

    with pytest.raises(ValueError):
        DocumentChunk(
            text="Sample content",
            chunk_index=0,
            page_number=None,
            section=None,
            token_count=2,
            source_block_indices=(),
        )
