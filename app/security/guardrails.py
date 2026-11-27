"""Validate question inputs before retrieval or generation."""

import re


INSTRUCTION_ATTACK = re.compile(
    # Catch common attempts to replace document-answering instructions.
    # Retrieved passage text is still treated as untrusted in the prompt.
    r"\b(ignore|disregard|override)\b.{0,50}\b(instructions|rules|system prompt)\b"
    r"|\b(reveal|print|show)\b.{0,50}\b(system prompt|api key|secret)\b",
    re.IGNORECASE | re.DOTALL,
)


def validate_question(question: str) -> str:
    """Return a bounded question, rejecting common instruction attacks."""
    if not isinstance(question, str):
        raise ValueError("Question must be text.")
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError("Question must contain 1 to 2000 characters.")
    if INSTRUCTION_ATTACK.search(question):
        raise ValueError("Question contains instructions unrelated to the documents.")
    return question
