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

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


_engine_kwargs = {"echo": False}
if "postgresql" in DATABASE_URL:
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_recycle=_int_env("DB_POOL_RECYCLE", 1800),
        pool_size=_int_env("DB_POOL_SIZE", 20),
        max_overflow=_int_env("DB_MAX_OVERFLOW", 20),
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

        def _migrate_jobs(sync_conn):
            insp = sa_inspect(sync_conn)
            if "jobs" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("jobs")}
                if "logs" not in cols:
                    sync_conn.execute(text("ALTER TABLE jobs ADD COLUMN logs JSON"))

        def _migrate_esign(sync_conn):
            insp = sa_inspect(sync_conn)
            tables = insp.get_table_names()
            # envelope_signers — new columns added across e-sign code review rounds
            if "envelope_signers" in tables:
                cols = {c["name"] for c in insp.get_columns("envelope_signers")}
                if "download_token_hash" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_signers ADD COLUMN download_token_hash VARCHAR(64)"
                    ))
                if "download_consumed_at" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_signers ADD COLUMN download_consumed_at TIMESTAMP"
                    ))
                if "last_viewed_at" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_signers ADD COLUMN last_viewed_at TIMESTAMP"
                    ))
            # envelope_audit_events — new forensic columns
            if "envelope_audit_events" in tables:
                cols = {c["name"] for c in insp.get_columns("envelope_audit_events")}
                if "notes" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_audit_events ADD COLUMN notes VARCHAR(500)"
                    ))
                if "user_agent" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_audit_events ADD COLUMN user_agent VARCHAR(1000)"
                    ))
            # envelope_batches — bulk-send columns added in templates/batches feature
            if "envelope_batches" in tables:
                cols = {c["name"] for c in insp.get_columns("envelope_batches")}
                if "column_map_json" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_batches ADD COLUMN column_map_json TEXT"
                    ))
                if "send_immediately" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE envelope_batches ADD COLUMN send_immediately BOOLEAN DEFAULT TRUE"
                    ))
            # signature_fields — bulk-apply group identifier (one drag → many cloned fields share a group)
            if "signature_fields" in tables:
                cols = {c["name"] for c in insp.get_columns("signature_fields")}
                if "bulk_group_id" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE signature_fields ADD COLUMN bulk_group_id VARCHAR(36)"
                    ))
                    sync_conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_signature_fields_bulk_group_id "
                        "ON signature_fields (bulk_group_id)"
                    ))

        def _migrate_skills(sync_conn):
            insp = sa_inspect(sync_conn)
            if "skills" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("skills")}
                if "requirements" not in cols:
                    sync_conn.execute(text("ALTER TABLE skills ADD COLUMN requirements JSON"))

        def _migrate_connections(sync_conn):
            insp = sa_inspect(sync_conn)
            if "connections" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("connections")}
                if "scope" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE connections ADD COLUMN scope VARCHAR(20) "
                        "NOT NULL DEFAULT 'private'"
                    ))
                    sync_conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_connections_scope_connector "
                        "ON connections (scope, connector_id)"
                    ))
                if "owner_org" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE connections ADD COLUMN owner_org VARCHAR(100)"
                    ))
                if "allowed_user_ids" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE connections ADD COLUMN allowed_user_ids JSON"
                    ))

        await conn.run_sync(_migrate_conversations)
        await conn.run_sync(_migrate_batch_jobs)
        await conn.run_sync(_migrate_users)
        await conn.run_sync(_migrate_jobs)
        await conn.run_sync(_migrate_esign)
        await conn.run_sync(_migrate_skills)
        await conn.run_sync(_migrate_connections)
        # Ensure new e-sign template + batch model classes are registered with Base.metadata
        # before create_all runs (importing the module registers the classes).
        from idpkit.esign import models as _esign_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_indexes)


def _migrate_indexes(sync_conn):
    """Idempotently create composite indexes on hot query paths.

    ``CREATE INDEX IF NOT EXISTS`` is supported by both PostgreSQL and SQLite,
    so existing deployments pick these up on next startup without an explicit
    migration step.
    """
    from sqlalchemy import text

    statements = [
        "CREATE INDEX IF NOT EXISTS ix_documents_owner_created "
        "ON documents (owner_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_conversations_owner_created "
        "ON conversations (owner_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_conv_messages_conv_created "
        "ON conversation_messages (conversation_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_batch_items_job_status "
        "ON batch_items (batch_job_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_conn_audit_conn_created "
        "ON connection_audit_log (connection_id, created_at)",
    ]
    import logging as _logging
    _log = _logging.getLogger(__name__)
    for stmt in statements:
        try:
            sync_conn.execute(text(stmt))
        except Exception as exc:
            # A pre-existing index with the same name on a different shape
            # shouldn't kill startup — log-and-continue is safer than failing,
            # but we log loudly so operators notice missing perf indexes.
            _log.warning("Index migration skipped (%s): %s", stmt, exc)


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
