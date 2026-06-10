"""Basic smoke tests for FarmGenius API and database."""

import os
import pytest

# Force SQLite for testing
os.environ["DATABASE_URL"] = "sqlite:///test_farmgenius.db"

from fastapi.testclient import TestClient
from src.app import app
from src.db.connection import engine, Base, SessionLocal
from src.db import crud as db_crud


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Cleanup test DB file
    if os.path.exists("test_farmgenius.db"):
        try:
            os.remove("test_farmgenius.db")
        except Exception:
            pass


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---- Health check ----
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ---- Root serves something ----
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


# ---- Database CRUD ----
def test_create_and_get_user(db):
    user = db_crud.get_or_create_user(db, "test_user_1")
    assert user.user_id == "test_user_1"
    assert user.language == "en"

    fetched = db_crud.get_user(db, "test_user_1")
    assert fetched is not None
    assert fetched.user_id == "test_user_1"


def test_chat_history(db):
    entry = db_crud.add_chat_entry(
        db,
        user_id="test_user_2",
        query="What is the price of wheat?",
        response="Wheat is $250/ton",
        intent="price",
        confidence=0.92,
        source="price_model",
    )
    assert entry.id is not None
    assert entry.query == "What is the price of wheat?"

    history = db_crud.get_chat_history(db, "test_user_2", limit=5)
    assert len(history) == 1
    assert history[0].intent == "price"


def test_user_statistics(db):
    db_crud.add_chat_entry(db, "stats_user", "q1", "r1", intent="weather")
    db_crud.add_chat_entry(db, "stats_user", "q2", "r2", intent="price")
    db_crud.add_chat_entry(db, "stats_user", "q3", "r3", intent="price")

    stats = db_crud.get_user_statistics(db, "stats_user")
    assert stats["success"] is True
    assert stats["total_queries"] == 3
    assert stats["queries_by_intent"]["price"] == 2
    assert stats["queries_by_intent"]["weather"] == 1


def test_update_preferences(db):
    user = db_crud.update_user_preferences(
        db, "pref_user", {"crops": ["wheat", "rice"], "region": "Punjab"}
    )
    assert user.preferences["crops"] == ["wheat", "rice"]

    user = db_crud.update_user_preferences(db, "pref_user", {"region": "Maharashtra"})
    assert user.preferences["region"] == "Maharashtra"
    assert user.preferences["crops"] == ["wheat", "rice"]  # preserved
