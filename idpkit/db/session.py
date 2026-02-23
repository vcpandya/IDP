"""IDP Kit database session management."""

import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from .models import Base

DATABASE_URL = os.getenv("IDP_DATABASE_URL", "sqlite+aiosqlite:///./idpkit.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables, migrating legacy schemas when needed."""
    from sqlalchemy import inspect as sa_inspect, text

    async with engine.begin() as conn:
        # Migrate conversation_messages: if the old table exists without the
        # owner_id column, drop it (and the new conversations table if somehow
        # partially created) so create_all rebuilds with the correct schema.
        def _migrate_conversations(sync_conn):
            insp = sa_inspect(sync_conn)
            if "conversation_messages" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("conversation_messages")}
                if "owner_id" not in cols:
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
