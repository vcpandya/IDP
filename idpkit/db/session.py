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

engine = create_async_engine(DATABASE_URL, echo=False)
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

        await conn.run_sync(_migrate_conversations)
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
