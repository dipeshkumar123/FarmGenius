"""
CRUD helpers for FarmGenius database operations.

All functions accept a SQLAlchemy Session instance.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.db.models import User, ChatHistory, QueryLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def get_or_create_user(db: Session, user_id: str, language: str = "en") -> User:
    """Return existing user or create a new one."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, language=language)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user: {user_id}")
    return user


def get_user(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Look up a user by email address."""
    return db.query(User).filter(User.email == email).first()


def create_registered_user(
    db: Session,
    user_id: str,
    email: str,
    display_name: str,
    password_hash: str,
    language: str = "en",
) -> User:
    """Create a fully-registered user with credentials."""
    user = User(
        user_id=user_id,
        email=email,
        display_name=display_name,
        password_hash=password_hash,
        language=language,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Registered new user: {user_id} ({email})")
    return user


def update_user_preferences(db: Session, user_id: str, preferences: Dict) -> User:
    user = get_or_create_user(db, user_id)
    current = user.preferences or {}
    current.update(preferences)
    user.preferences = current
    user.last_active = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


def update_user_language(db: Session, user_id: str, language: str) -> User:
    user = get_or_create_user(db, user_id)
    user.language = language
    user.last_active = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def add_chat_entry(
    db: Session,
    user_id: str,
    query: str,
    response: str,
    intent: str = None,
    confidence: float = None,
    source: str = None,
) -> ChatHistory:
    """Append a chat exchange to history (auto-creates user if needed)."""
    get_or_create_user(db, user_id)
    entry = ChatHistory(
        user_id=user_id,
        query=query,
        response=response,
        intent=intent,
        confidence=confidence,
        source=source,
    )
    db.add(entry)
    # Touch user last_active
    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        user.last_active = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry


def get_chat_history(db: Session, user_id: str, limit: int = 20) -> List[ChatHistory]:
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(desc(ChatHistory.timestamp))
        .limit(limit)
        .all()
    )


def clear_chat_history(db: Session, user_id: str) -> int:
    """Delete all chat history for a user. Returns count of deleted rows."""
    count = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
    db.commit()
    return count


# ---------------------------------------------------------------------------
# Query log
# ---------------------------------------------------------------------------

def log_query(
    db: Session,
    user_id: str,
    query_type: str,
    query_text: str = None,
    response_summary: str = None,
    metadata: Dict = None,
) -> QueryLog:
    get_or_create_user(db, user_id)
    entry = QueryLog(
        user_id=user_id,
        query_type=query_type,
        query_text=query_text,
        response_summary=response_summary,
        metadata_=metadata,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_user_statistics(db: Session, user_id: str) -> Dict[str, Any]:
    """Aggregate user activity statistics."""
    user = get_user(db, user_id)
    if not user:
        return {"success": False, "message": f"User {user_id} not found"}

    total = db.query(func.count(ChatHistory.id)).filter(ChatHistory.user_id == user_id).scalar() or 0

    # Count by intent
    intent_counts = (
        db.query(ChatHistory.intent, func.count(ChatHistory.id))
        .filter(ChatHistory.user_id == user_id)
        .group_by(ChatHistory.intent)
        .all()
    )
    by_intent = {intent or "unknown": count for intent, count in intent_counts}

    return {
        "success": True,
        "user_id": user_id,
        "total_queries": total,
        "queries_by_intent": by_intent,
        "last_active": user.last_active.isoformat() if user.last_active else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "message": "Statistics retrieved successfully",
    }
