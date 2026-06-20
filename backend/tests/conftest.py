import pytest
from app.api.middleware import rate_limit_store

from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="function", autouse=True)
def clear_rate_limit():
    """Clear the rate limit store before each test to prevent 429 cascades across files."""
    rate_limit_store.clear()

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Shared TestClient for the entire test session."""
    with TestClient(app) as c:
        yield c
