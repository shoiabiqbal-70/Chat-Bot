"""Security checks for public accounts and document isolation."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from fastapi import Request, Response

from app.api import documents
from app.api import auth
from app.api.auth import get_current_user_id, hash_password, verify_password
from app.config import settings
from app.db.session import get_db
from app.main import app
from app.services.answer_service import answer_question
from app.models import User


def test_password_hash_uses_a_unique_salt():
    first = hash_password("long-demo-password")
    second = hash_password("long-demo-password")
    assert first != second
    assert verify_password("long-demo-password", first)
    assert not verify_password("wrong-password", first)


def test_session_cookie_is_secure_and_http_only(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")

    class Session:
        def add(self, value):
            self.record = value

        async def commit(self):
            pass

    response = Response()
    session = Session()
    user = SimpleNamespace(id=uuid4(), email="person@example.com")
    account = asyncio.run(auth.issue_session(response, session, user))
    cookie = response.headers["set-cookie"]
    assert account.email == user.email
    assert "__Host-source_atlas_session=" in cookie
    assert "httponly" in cookie.lower()
    assert "secure" in cookie.lower()
    assert "samesite=strict" in cookie.lower()
    assert session.record.user_id == user.id


def test_signup_and_signin_routes(monkeypatch):
    class Session:
        user = None

        def add(self, value):
            if isinstance(value, User):
                self.user = value

        async def flush(self):
            self.user.id = uuid4()

        async def commit(self):
            pass

        async def execute(self, statement):
            return SimpleNamespace(scalar_one_or_none=lambda: self.user)

    session = Session()

    async def fake_db():
        yield session

    app.dependency_overrides[get_db] = fake_db
    try:
        with TestClient(app) as client:
            credentials = {"email": "Person@Example.com", "password": "a-unique-demo-password"}
            created = client.post("/api/auth/signup", json=credentials)
            assert created.status_code == 201
            assert created.json()["email"] == "person@example.com"
            assert "httponly" in created.headers["set-cookie"].lower()
            signed_in = client.post("/api/auth/signin", json=credentials)
            assert signed_in.status_code == 200
            assert signed_in.json()["id"] == created.json()["id"]
    finally:
        app.dependency_overrides.clear()


def test_other_account_cannot_read_or_delete_document(monkeypatch):
    mine, other = uuid4(), uuid4()
    document = SimpleNamespace(id=uuid4(), owner_id=other)

    class Repository:
        def __init__(self, session):
            pass

        async def get_by_id(self, document_id):
            return document

    async def fake_db():
        yield object()

    async def fake_user():
        return mine

    monkeypatch.setattr(documents, "DocumentRepository", Repository)
    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_current_user_id] = fake_user
    try:
        with TestClient(app) as client:
            assert client.get(f"/api/documents/{document.id}").status_code == 404
            assert client.delete(f"/api/documents/{document.id}").status_code == 404
            assert client.post(f"/api/documents/{document.id}/reindex").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_selected_document_rejects_other_owner(monkeypatch):
    from app.services import answer_service

    mine, other = uuid4(), uuid4()
    document_id = uuid4()

    class Repository:
        def __init__(self, session):
            pass

        async def get_by_id(self, value):
            return SimpleNamespace(id=document_id, status="completed", owner_id=other)

    monkeypatch.setattr(answer_service, "DocumentRepository", Repository)
    with pytest.raises(ValueError, match="not available"):
        asyncio.run(answer_question("What is the title?", object(), object(),
                                    document_id=document_id, owner_id=mine))
