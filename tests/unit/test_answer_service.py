"""Grounding and abstention checks for the answer service."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.services import answer_service


class FakeEmbedding:
    def embed_query(self, question):
        return SimpleNamespace(tolist=lambda: [1.0, 0.0])


class FakeVectors:
    def __init__(self, matches):
        self.matches = matches

    async def query(self, vector, top_k, document_id=None):
        return self.matches


def test_no_matches_abstains(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    result = asyncio.run(answer_service.answer_question(
        "What is energy?", object(), FakeVectors([])
    ))
    assert result.abstained
    assert result.citations == []
    assert result.chunk_count == 0


def test_low_similarity_abstains_without_generation(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    result = asyncio.run(answer_service.answer_question(
        "Unrelated question", object(),
        FakeVectors([SimpleNamespace(id=str(uuid4()), score=0.1)])
    ))
    assert result.abstained
    assert result.citations == []
    assert result.chunk_count == 0


def test_rejects_citation_outside_retrieval(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    monkeypatch.setattr(answer_service.settings, "groq_api_key", SecretStr("test-key"))
    document_id, chunk_id = uuid4(), uuid4()
    chunk = SimpleNamespace(id=chunk_id, document_id=document_id, text="Energy is conserved.",
                            page_number=2, chunk_index=0)
    document = SimpleNamespace(id=document_id, filename="guide.pdf", status="completed")

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return chunk

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return document

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)
    bad_id = uuid4()

    def respond(request):
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content)["model"] == "openai/gpt-oss-20b"
        assert json.loads(request.content)["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '{"answer":"Invented","abstained":false,"citations":["'
                       + str(bad_id) + '"]}'
        }}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await answer_service.answer_question(
                "What is energy?", object(),
                FakeVectors([SimpleNamespace(id=str(chunk_id), score=0.9)]), client
            )

    with pytest.raises(answer_service.AnswerError):
        asyncio.run(run())


def test_provider_access_denial_has_actionable_message(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    monkeypatch.setattr(answer_service.settings, "groq_api_key", SecretStr("test-key"))
    document_id, chunk_id = uuid4(), uuid4()
    chunk = SimpleNamespace(id=chunk_id, document_id=document_id,
                            text="Energy is conserved.", page_number=2, chunk_index=0)
    document = SimpleNamespace(id=document_id, filename="guide.pdf", status="completed")

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return chunk

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return document

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(403, json={"error": {"message": "Forbidden"}})
        )) as client:
            await answer_service.answer_question(
                "What is energy?", object(),
                FakeVectors([SimpleNamespace(id=str(chunk_id), score=0.9)]), client
            )

    with pytest.raises(answer_service.AnswerError, match="Groq denied API access \\(HTTP 403\\)"):
        asyncio.run(run())


def test_selected_document_journal_question_includes_opening_chunk(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    monkeypatch.setattr(answer_service.settings, "groq_api_key", SecretStr("test-key"))
    document_id, opening_id, other_id = uuid4(), uuid4(), uuid4()
    opening = SimpleNamespace(id=opening_id, document_id=document_id,
                              text="Computers & Operations Research journal homepage",
                              page_number=1, chunk_index=0)
    other = SimpleNamespace(id=other_id, document_id=document_id,
                            text="Journal citations in related work", page_number=13,
                            chunk_index=80)
    document = SimpleNamespace(id=document_id, filename="article.pdf", status="completed")

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return other

        async def list_for_document(self, value, limit, offset=0):
            assert value == document_id and offset == 0
            return [opening]

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return document

    class FakeVectors:
        async def query(self, vector, top_k, selected_id):
            assert selected_id == str(document_id)
            return [SimpleNamespace(id=str(other_id), score=0.8)]

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)

    def respond(request):
        payload = json.loads(request.content)
        assert str(opening_id) in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"answer": "Computers & Operations Research",
                                   "abstained": False, "citations": [str(opening_id)]})
        }}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await answer_service.answer_question(
                "What is the journal name?", object(), FakeVectors(), client,
                document_id=document_id)

    result = asyncio.run(run())
    assert result.answer == "Computers & Operations Research"
    assert result.citations == [opening_id]
    assert result.chunk_count == 2


def test_selected_document_recovers_keyword_passage_when_vector_misses(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    monkeypatch.setattr(answer_service.settings, "groq_api_key", SecretStr("test-key"))
    document_id, chunk_id = uuid4(), uuid4()
    chunk = SimpleNamespace(id=chunk_id, document_id=document_id,
                            text="The vehicle routing method uses clique search.",
                            page_number=4, chunk_index=7)
    rows = [SimpleNamespace(id=uuid4(), document_id=document_id,
                            text="Background passage about another topic.",
                            page_number=1, chunk_index=index)
            for index in range(12)] + [chunk]
    document = SimpleNamespace(id=document_id, filename="paper.pdf", status="completed")

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def list_for_document(self, value, limit, offset):
            return rows[offset:offset + limit]

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return document

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)

    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"answer": "It uses clique search.",
                                   "abstained": False, "citations": [str(chunk_id)]})
        }}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await answer_service.answer_question(
                "What vehicle routing method does the paper use?", object(),
                FakeVectors([]), client, document_id=document_id)

    result = asyncio.run(run())
    assert result.answer == "It uses clique search."
    assert result.citations == [chunk_id]


def test_short_selected_document_checks_all_passages(monkeypatch):
    monkeypatch.setattr(answer_service, "get_embedding_service", lambda: FakeEmbedding())
    monkeypatch.setattr(answer_service.settings, "groq_api_key", SecretStr("test-key"))
    document_id, first_id, second_id = uuid4(), uuid4(), uuid4()
    rows = [SimpleNamespace(id=first_id, document_id=document_id,
                            text="General introduction.", page_number=1, chunk_index=0),
            SimpleNamespace(id=second_id, document_id=document_id,
                            text="A relevant fact appears here.", page_number=2, chunk_index=1)]

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def list_for_document(self, value, limit, offset):
            return rows[offset:offset + limit]

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return SimpleNamespace(id=document_id, filename="short.pdf", status="completed")

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)

    def respond(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        assert str(first_id) in prompt and str(second_id) in prompt
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"answer": "A relevant fact appears here.",
                                   "abstained": False, "citations": [str(second_id)]})
        }}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await answer_service.answer_question(
                "Explain the relevant fact", object(), FakeVectors([]), client,
                document_id=document_id)

    result = asyncio.run(run())
    assert result.citations == [second_id]


@pytest.mark.parametrize(("question", "texts", "expected"), [
    ("How many figures in this PDF?",
     ["Fig. 1 describes the model. Fig. 1 is discussed again.",
      "Fig. 2 shows results. Fig. 3 shows a comparison."],
     "3 numbered figures"),
    ("How many tables have in this PDF?",
     ["Table 1 lists instances. Table 2 shows results.",
      "Table 3 compares methods. Table 3 is discussed again."],
     "3 numbered tables"),
])
def test_numbered_count_scans_selected_document_without_provider(
        monkeypatch, question, texts, expected):
    document_id = uuid4()
    rows = [SimpleNamespace(id=uuid4(), document_id=document_id,
                            text=text, page_number=index + 1, chunk_index=index)
            for index, text in enumerate(texts)]

    class FakeChunkRepository:
        def __init__(self, session):
            pass

        async def list_for_document(self, value, limit, offset):
            assert value == document_id
            return rows[offset:offset + limit]

    class FakeDocumentRepository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return SimpleNamespace(id=document_id, filename="paper.pdf", status="completed")

    monkeypatch.setattr(answer_service, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(answer_service, "DocumentRepository", FakeDocumentRepository)
    result = asyncio.run(answer_service.answer_question(
        question, object(), object(), document_id=document_id))
    assert expected in result.answer
    assert not result.abstained
    assert set(result.citations) == {row.id for row in rows}


@pytest.mark.parametrize("question", ["How many figures in this PDF?",
                                       "How many tables in this PDF?"])
def test_numbered_count_requires_selected_document(question):
    with pytest.raises(ValueError, match="Select a document"):
        asyncio.run(answer_service.answer_question(
            question, object(), object()))
