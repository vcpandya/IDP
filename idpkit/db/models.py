"""IDP Kit database models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


def generate_uuid():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default=UserRole.USER.value)
    api_key = Column(String(64), unique=True, nullable=True, index=True)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=utcnow)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    prompts = relationship("Prompt", back_populates="owner", cascade="all, delete-orphan")
    templates = relationship("Template", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String(500), nullable=False)
    format = Column(String(20))  # pdf, docx, md, html, xlsx, csv, pptx, image
    file_path = Column(String(1000))
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    status = Column(String(20), default="uploaded")  # uploaded, indexing, indexed, failed
    tree_index = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="documents")
    jobs = relationship("Job", back_populates="document", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    job_type = Column(String(50), nullable=False)  # index, extract, convert, tool, batch
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    params = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="jobs")


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="prompts")


class Template(Base):
    __tablename__ = "templates"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    format = Column(String(20))  # docx, md
    file_path = Column(String(1000), nullable=True)
    schema_json = Column(JSON, nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="templates")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, tool, system
    content = Column(Text, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    tool_name = Column(String(100), nullable=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)


# Import graph models so Base.metadata.create_all() picks up their tables.
# Placed at the bottom to avoid circular imports (graph.models imports from here).
import idpkit.graph.models as _graph_models  # noqa: F401, E402
