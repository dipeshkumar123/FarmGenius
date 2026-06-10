"""
Database connection setup for FarmGenius.

Supports:
  - PostgreSQL via DATABASE_URL env var  (production / Neon)
  - SQLite fallback via SQLITE_URL or default file  (local dev)

Usage:
    from src.db.connection import get_db, engine
"""

import os
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve database URL
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Vercel / Neon sometimes gives "postgres://…" – SQLAlchemy requires "postgresql://…"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Fallback to SQLite for local development
if not DATABASE_URL:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _db_path = os.path.join(_project_root, "data", "farmgenius.db")
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    DATABASE_URL = f"sqlite:///{_db_path}"
    logger.info(f"No DATABASE_URL set — using SQLite: {_db_path}")

# ---------------------------------------------------------------------------
# Engine & Session factories
# ---------------------------------------------------------------------------
_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,  # reconnect stale connections
)

# Enable WAL mode for SQLite (better concurrency)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Yield a database session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    from src.db.models import User, ChatHistory, QueryLog  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")
