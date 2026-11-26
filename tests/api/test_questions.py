"""Question API error contract."""

from fastapi.testclient import TestClient

from app.api import questions
from app.db.session import get_db
from app.main import app
from app.services.answer_service import RateLimitError


def test_answer_provider_rate_limit_returns_429(monkeypatch):
    async def fake_db():
        yield object()

    async def fake_answer(*args, **kwargs):
        raise RateLimitError("Answer provider rate limit exceeded.")

    monkeypatch.setattr(questions, "VectorStore", lambda: object())
    monkeypatch.setattr(questions, "answer_question", fake_answer)
    app.dependency_overrides[get_db] = fake_db
    try:
        with TestClient(app) as client:
            response = client.post("/api/ask", json={"question": "What is energy?"})
        assert response.status_code == 429
    finally:
        app.dependency_overrides.pop(get_db, None)
