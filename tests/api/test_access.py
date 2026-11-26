"""Public demo exposes the sign-in page while protecting account data."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_public_workspace_and_health(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/app").status_code == 200
        assert client.get("/api/documents").status_code == 401
        assert client.post("/api/ask", json={"question": "Hello"}).status_code == 401


def test_production_rejects_missing_provider_config(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "groq_api_key", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        with TestClient(app):
            pass
