"""
SQLAlchemy ORM models for FarmGenius.

Tables:
  - users:        user profiles, preferences, language
  - chat_history: per-user conversation log
  - query_log:    typed query log for analytics
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from src.db.connection import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(String(128), primary_key=True, index=True)
    email = Column(String(256), unique=True, nullable=True, index=True)
    display_name = Column(String(128), nullable=True)
    password_hash = Column(String(256), nullable=True)  # None for legacy/guest users
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    preferences = Column(JSON, default=dict)
    language = Column(String(10), default="en")

    # Relationships
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "email": self.email,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "preferences": self.preferences or {},
            "language": self.language,
        }


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    intent = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    source = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="chat_history")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query": self.query,
            "response": self.response,
            "intent": self.intent,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class QueryLog(Base):
    __tablename__ = "query_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    query_type = Column(String(32), nullable=False)  # chat, crop, weather, price, disease
    query_text = Column(Text, nullable=True)
    response_summary = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="query_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query_type": self.query_type,
            "query_text": self.query_text,
            "response_summary": self.response_summary,
            "metadata": self.metadata_,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
