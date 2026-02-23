"""IDP Kit database models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import DateTime as _SADateTime, TypeDecorator
import enum


class TZDateTime(TypeDecorator):
    """A DateTime type that stores timezone-aware datetimes in PostgreSQL
    and handles naive datetimes for SQLite compatibility."""
    impl = _SADateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


def generate_uuid():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


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
    created_at = Column(TZDateTime, default=utcnow)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    tags = relationship("Tag", back_populates="owner", cascade="all, delete-orphan")
    prompts = relationship("Prompt", back_populates="owner", cascade="all, delete-orphan")
    templates = relationship("Template", back_populates="owner", cascade="all, delete-orphan")
    processing_templates = relationship("ProcessingTemplate", back_populates="owner", cascade="all, delete-orphan")
    batch_jobs = relationship("BatchJob", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")


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
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="documents")
    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
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
    created_at = Column(TZDateTime, default=utcnow)
    completed_at = Column(TZDateTime, nullable=True)

    document = relationship("Document", back_populates="jobs")


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

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
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="templates")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String(200), nullable=False, default="New conversation")
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="conversations")
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(
        String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, tool, system
    content = Column(Text, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    tool_name = Column(String(100), nullable=True)
    sources_json = Column(JSON, nullable=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    created_at = Column(TZDateTime, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class ProcessingTemplate(Base):
    """Reusable batch processing template — prompt + tool config + output formatting."""

    __tablename__ = "processing_templates"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=True)
    tool_name = Column(String(100), nullable=False, default="custom")
    tool_options = Column(JSON, nullable=True)
    reference_content = Column(Text, nullable=True)
    output_format = Column(String(20), default="json")
    output_template = Column(JSON, nullable=True)
    model = Column(String(200), nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    is_public = Column(Integer, default=0)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="processing_templates")
    batch_jobs = relationship("BatchJob", back_populates="template")


class BatchJob(Base):
    """Parent record tracking a batch of document processing operations."""

    __tablename__ = "batch_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    template_id = Column(String, ForeignKey("processing_templates.id"), nullable=True)
    tool_name = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")
    progress = Column(Integer, default=0)
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    prompt = Column(Text, nullable=True)
    options = Column(JSON, nullable=True)
    model = Column(String(200), nullable=True)
    concurrency = Column(Integer, default=3)
    result_summary = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=utcnow)
    started_at = Column(TZDateTime, nullable=True)
    completed_at = Column(TZDateTime, nullable=True)

    owner = relationship("User", back_populates="batch_jobs")
    template = relationship("ProcessingTemplate", back_populates="batch_jobs")
    items = relationship("BatchItem", back_populates="batch_job", cascade="all, delete-orphan")


class BatchItem(Base):
    """Per-document child record within a batch."""

    __tablename__ = "batch_items"

    id = Column(String, primary_key=True, default=generate_uuid)
    batch_job_id = Column(String, ForeignKey("batch_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    position = Column(Integer, default=0)
    status = Column(String(20), default="pending")
    result = Column(JSON, nullable=True)
    output_file = Column(String(1000), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=utcnow)
    started_at = Column(TZDateTime, nullable=True)
    completed_at = Column(TZDateTime, nullable=True)

    batch_job = relationship("BatchJob", back_populates="items")
    document = relationship("Document")


class Tag(Base):
    """A tag for grouping documents into logical collections."""

    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    color = Column(String(7), default="#4f46e5")
    description = Column(String(500), nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", back_populates="tags")
    documents = relationship("Document", secondary=document_tags, back_populates="tags")


# Import graph models so Base.metadata.create_all() picks up their tables.
# Placed at the bottom to avoid circular imports (graph.models imports from here).
import idpkit.graph.models as _graph_models  # noqa: F401, E402
