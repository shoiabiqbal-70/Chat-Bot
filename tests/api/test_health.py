"""Tests for the health-check endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_success() -> None:
    """Verify that the health endpoint returns HTTP 200."""

    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["app_name"] == "Production RAG Service"
    assert data["environment"] == "development"


def test_root_endpoint_returns_welcome_message() -> None:
    """Verify that the root endpoint returns application information."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"
