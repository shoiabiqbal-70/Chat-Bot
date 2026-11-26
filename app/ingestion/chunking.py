
"""Token-aware document chunking with citation preservation."""

from dataclasses import dataclass
import re
from typing import Protocol

from app.ingestion.extraction_models import ExtractedDocument


DEFAULT_CHUNK_SIZE = 320
DEFAULT_CHUNK_OVERLAP = 45


class TokenizerProtocol(Protocol):
    """Define the tokenizer interface required by the chunker."""

    def encode(self, text: str):
        ...

    def decode(self, ids: list[int]) -> str:
        ...


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Represent a searchable chunk with source metadata."""

    text: str
    chunk_index: int
    page_number: int | None
    section: str | None
    token_count: int
    source_block_indices: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Chunk text cannot be empty.")

        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative.")

        if type(self.token_count) is not int or self.token_count < 1:
            raise ValueError("Token count must be positive.")

        if not isinstance(self.source_block_indices, tuple):
            raise ValueError("Source block indexes must be a tuple.")

        if not self.source_block_indices:
            raise ValueError("Source block indexes are required.")

        if any(
            type(index) is not int or index < 0
            for index in self.source_block_indices
        ):
            raise ValueError("Invalid source block indexes.")

        if len(set(self.source_block_indices)) != len(
            self.source_block_indices
        ):
            raise ValueError("Duplicate source block indexes.")

        if (
            self.page_number is not None
            and (
                type(self.page_number) is not int
                or self.page_number < 1
            )
        ):
            raise ValueError("Page number must be positive.")

        if self.section is not None and not isinstance(
            self.section, str
        ):
            raise ValueError("Section must be a string or None.")


def _validate_settings(chunk_size, overlap):
    """Validate chunking configuration."""

    if type(chunk_size) is not int or chunk_size < 2:
        raise ValueError(
            "Chunk size must be an integer of at least 2."
        )

    if type(overlap) is not int or overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )


def _token_ids(tokenizer, text):
    """Return tokenizer IDs for text."""

    encoded = tokenizer.encode(text)

    ids = getattr(encoded, "ids", None)

    if not isinstance(ids, list):
        raise TypeError(
            "Tokenizer encode() must return an object with an IDs list."
        )

    if any(type(token_id) is not int for token_id in ids):
        raise TypeError("Tokenizer returned invalid token IDs.")

    return ids


def _decode(tokenizer, ids):
    """Decode token IDs into readable text."""

    decoded = tokenizer.decode(ids)

    if not isinstance(decoded, str):
        raise TypeError("Tokenizer decode() must return a string.")

    return decoded.strip()


def _sentence_units(text):
    """Split text into sentence-aware units."""

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_document(
    document: ExtractedDocument,
    tokenizer: TokenizerProtocol,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Convert extracted document blocks into searchable chunks.

    Preserves:
        - Original page numbers.
        - Original document sections.
        - Source block indexes.
        - Sequential chunk indexes.
        - Configured token overlap.

    Overlap never crosses page or section boundaries.

    Long sentences are split into token windows when necessary.
    """

    _validate_settings(chunk_size, overlap)

    if not isinstance(document, ExtractedDocument):
        raise TypeError(
            "Expected an ExtractedDocument instance."
        )

    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError(
            "Tokenizer must provide encode()."
        )

    if not callable(getattr(tokenizer, "decode", None)):
        raise TypeError(
            "Tokenizer must provide decode()."
        )

    if not document.blocks:
        return []

    # Each entry contains (token_id, source_block_index).
    # This preserves exact provenance even for overlap tokens.
    groups = []

    for block in document.blocks:

        key = (
            block.page_number,
            block.section,
        )

        if not groups or groups[-1]["key"] != key:
            groups.append({
                "key": key,
                "units": [],
            })

        group = groups[-1]

        for sentence in _sentence_units(block.text):
            ids = _token_ids(tokenizer, sentence)

            if ids:
                group["units"].append(
                    (ids, block.block_index)
                )

    results: list[DocumentChunk] = []

    def emit(entries, page_number, section):
        """Create one chunk with exact source provenance."""

        if not entries:
            return

        ids = [token_id for token_id, _ in entries]

        text = _decode(tokenizer, ids)

        if not text:
            raise ValueError(
                "Tokenizer produced an empty decoded chunk."
            )

        source_indexes = tuple(sorted({
            source_index
            for _, source_index in entries
        }))

        results.append(
            DocumentChunk(
                text=text,
                chunk_index=len(results),
                page_number=page_number,
                section=section,
                token_count=len(ids),
                source_block_indices=source_indexes,
            )
        )

    # Process each page/section independently.
    for group in groups:

        page_number, section = group["key"]

        current_entries = []

        for sentence_ids, source_index in group["units"]:

            sentence_entries = [
                (token_id, source_index)
                for token_id in sentence_ids
            ]

            # Handle unusually long sentences separately.
            if len(sentence_entries) > chunk_size:

                if current_entries:
                    emit(
                        current_entries,
                        page_number,
                        section,
                    )

                current_entries = []

                step = chunk_size - overlap
                position = 0

                while position < len(sentence_entries):

                    window = sentence_entries[
                        position:position + chunk_size
                    ]

                    emit(
                        window,
                        page_number,
                        section,
                    )

                    if (
                        position + chunk_size
                        >= len(sentence_entries)
                    ):
                        break

                    position += step

                continue

            # Include a separator when joining sentence units.
            separator_ids = (
                _token_ids(tokenizer, " ")
                if current_entries
                else []
            )

            addition = [
                (token_id, source_index)
                for token_id in separator_ids + sentence_ids
            ]

            # If the next sentence does not fit, emit the
            # current chunk and retain its exact token provenance.
            if (
                current_entries
                and len(current_entries) + len(addition) > chunk_size
            ):

                previous_entries = current_entries

                emit(
                    previous_entries,
                    page_number,
                    section,
                )

                current_entries = (
                    previous_entries[-overlap:]
                    if overlap
                    else []
                )

                # Check whether overlap plus the next sentence fits.
                separator_ids = (
                    _token_ids(tokenizer, " ")
                    if current_entries
                    else []
                )

                addition = [
                    (token_id, source_index)
                    for token_id in separator_ids + sentence_ids
                ]

                # Discard overlap if it prevents the sentence
                # from fitting. Never create overlap-only chunks.
                if len(current_entries) + len(addition) > chunk_size:
                    current_entries = []

                    addition = [
                        (token_id, source_index)
                        for token_id in sentence_ids
                    ]

            if len(current_entries) + len(addition) > chunk_size:
                raise ValueError(
                    "Unable to fit sentence into configured chunk size."
                )

            current_entries.extend(addition)

        # Finish the current page/section without carrying
        # overlap into another page or section.
        if current_entries:
            emit(
                current_entries,
                page_number,
                section,
            )

    return results
