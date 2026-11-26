"""The hosted workspace requires credentials before exposing documents."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import settings
from app.main import app


def test_workspace_password_gate(monkeypatch):
    monkeypatch.setattr(settings, "app_access_password", SecretStr("private-test-password"))
    monkeypatch.setattr(settings, "app_access_username", "owner")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        denied = client.get("/app")
        assert denied.status_code == 401
        assert "Basic" in denied.headers["www-authenticate"]
        assert client.get("/api/documents", auth=("owner", "wrong")).status_code == 401
        assert client.get("/app", auth=("owner", "private-test-password")).status_code == 200


def test_production_rejects_missing_password(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "app_access_password", None)
    with pytest.raises(RuntimeError, match="APP_ACCESS_PASSWORD"):
        with TestClient(app):
            pass
