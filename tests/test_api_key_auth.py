"""
Tests for Bearer token authentication on protected API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from src.core.config import settings

TEST_API_KEY = "test_key_12345"


@pytest.fixture(autouse=True)
def require_api_key():
    previous_api_key = settings.API_KEY
    previous_require_api_key = settings.REQUIRE_API_KEY
    settings.API_KEY = TEST_API_KEY
    settings.REQUIRE_API_KEY = True
    yield
    settings.API_KEY = previous_api_key
    settings.REQUIRE_API_KEY = previous_require_api_key


@pytest.fixture
def client():
    return TestClient(app)


def test_protected_endpoint_requires_bearer_token(client):
    response = client.post("/extract")

    assert response.status_code == 401
    assert "Bearer token is required" in response.json()["detail"]


def test_protected_endpoint_rejects_invalid_bearer_token(client):
    response = client.post(
        "/extract",
        headers={"Authorization": "Bearer wrong_key"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid bearer token"


def test_protected_endpoint_accepts_valid_bearer_token(client):
    response = client.post(
        "/extract",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )

    # Authentication passed; the request then fails validation because no PDF
    # was provided.
    assert response.status_code == 422


def test_health_endpoint_remains_public_for_container_probes(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "Michman PDF Extractor"
