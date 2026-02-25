"""IDP Kit database session management."""

import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from .models import Base


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("IDP_DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if "sslmode" in url:
            url = url.split("?")[0]
        return url
    return "sqlite+aiosqlite:///./idpkit.db"


DATABASE_URL = _get_database_url()

_engine_kwargs = {"echo": False}
if "postgresql" in DATABASE_URL:
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=3,
        max_overflow=5,
    )

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables, migrating legacy schemas when needed."""
    from sqlalchemy import inspect as sa_inspect, text

    async with engine.begin() as conn:
        def _migrate_conversations(sync_conn):
            insp = sa_inspect(sync_conn)
            if "conversation_messages" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("conversation_messages")}
                if "owner_id" not in cols or "source_type" not in cols:
                    sync_conn.execute(text("DROP TABLE IF EXISTS conversation_messages"))
                    sync_conn.execute(text("DROP TABLE IF EXISTS conversations"))

        def _migrate_batch_jobs(sync_conn):
            insp = sa_inspect(sync_conn)
            if "batch_jobs" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("batch_jobs")}
                if "reference_doc_ids" not in cols:
                    sync_conn.execute(text("ALTER TABLE batch_jobs ADD COLUMN reference_doc_ids JSON"))
                if "generated_schema" not in cols:
                    sync_conn.execute(text("ALTER TABLE batch_jobs ADD COLUMN generated_schema JSON"))

        def _migrate_users(sync_conn):
            insp = sa_inspect(sync_conn)
            if "users" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("users")}
                if "default_provider" not in cols:
                    sync_conn.execute(text("ALTER TABLE users ADD COLUMN default_provider VARCHAR(50)"))
                if "default_model" not in cols:
                    sync_conn.execute(text("ALTER TABLE users ADD COLUMN default_model VARCHAR(200)"))

        await conn.run_sync(_migrate_conversations)
        await conn.run_sync(_migrate_batch_jobs)
        await conn.run_sync(_migrate_users)
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency that provides a database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
