"""IDP Kit database session management."""

import os
from sqlalchemy import inspect as sa_inspect, text
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
                elif "computations_json" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE conversation_messages "
                        "ADD COLUMN computations_json JSON"
                    ))

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

        def _migrate_documents(sync_conn):
            insp = sa_inspect(sync_conn)
            if "documents" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("documents")}
                if "doc_category" not in cols:
                    sync_conn.execute(text("ALTER TABLE documents ADD COLUMN doc_category VARCHAR(100)"))
                    sync_conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_documents_doc_category "
                        "ON documents (doc_category)"
                    ))
                if "doc_category_confidence" not in cols:
                    sync_conn.execute(text("ALTER TABLE documents ADD COLUMN doc_category_confidence INTEGER"))
                if "smart_metadata" not in cols:
                    sync_conn.execute(text("ALTER TABLE documents ADD COLUMN smart_metadata JSON"))

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

        def _migrate_metadata_jobs(sync_conn):
            insp = sa_inspect(sync_conn)
            if "metadata_jobs" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("metadata_jobs")}
                if "failures" not in cols:
                    sync_conn.execute(text(
                        "ALTER TABLE metadata_jobs ADD COLUMN failures TEXT"
                    ))

        await conn.run_sync(_migrate_conversations)
        await conn.run_sync(_migrate_batch_jobs)
        await conn.run_sync(_migrate_users)
        await conn.run_sync(_migrate_jobs)
        await conn.run_sync(_migrate_esign)
        await conn.run_sync(_migrate_skills)
        await conn.run_sync(_migrate_connections)
        await conn.run_sync(_migrate_documents)
        await conn.run_sync(_migrate_metadata_jobs)
        await conn.run_sync(_migrate_dedupe_tags)
        # Ensure new e-sign template + batch model classes are registered with Base.metadata
        # before create_all runs (importing the module registers the classes).
        from idpkit.esign import models as _esign_models  # noqa: F401
        from idpkit.metadata import models as _metadata_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_indexes)


def _migrate_dedupe_tags(sync_conn):
    """Consolidate duplicate per-owner tags, keeping the earliest of each name.

    Older deployments could create multiple tags with the same (case-insensitive)
    name for one owner — chiefly from Auto-Tag racing the create path. New
    duplicates are now prevented at the application layer (see
    :func:`lock_tag_name`); this migration cleans up any legacy duplicates on
    startup by merging each duplicate group into its earliest member: repoint
    document/conversation links to the keeper (avoiding duplicate links) then
    delete the losers. Idempotent — a deployment with no duplicates is a no-op.
    """
    from sqlalchemy import inspect as sa_inspect, text
    import logging as _logging

    _log = _logging.getLogger(__name__)
    insp = sa_inspect(sync_conn)
    names = set(insp.get_table_names())
    if "tags" not in names:
        return
    has_conv_tags = "conversation_tags" in names
    has_doc_tags = "document_tags" in names

    try:
        groups = sync_conn.execute(
            text(
                "SELECT owner_id, lower(name) AS lname FROM tags "
                "GROUP BY owner_id, lower(name) HAVING COUNT(*) > 1"
            )
        ).fetchall()
        for owner_id, lname in groups:
            members = sync_conn.execute(
                text(
                    "SELECT id FROM tags WHERE owner_id = :o AND lower(name) = :n "
                    "ORDER BY created_at ASC, id ASC"
                ),
                {"o": owner_id, "n": lname},
            ).fetchall()
            ids = [m[0] for m in members]
            if len(ids) < 2:
                continue
            keeper, losers = ids[0], ids[1:]
            for loser in losers:
                if has_doc_tags:
                    sync_conn.execute(
                        text(
                            "UPDATE document_tags SET tag_id = :k WHERE tag_id = :l "
                            "AND document_id NOT IN ("
                            "SELECT document_id FROM document_tags WHERE tag_id = :k)"
                        ),
                        {"k": keeper, "l": loser},
                    )
                    sync_conn.execute(
                        text("DELETE FROM document_tags WHERE tag_id = :l"),
                        {"l": loser},
                    )
                if has_conv_tags:
                    sync_conn.execute(
                        text(
                            "UPDATE conversation_tags SET tag_id = :k WHERE tag_id = :l "
                            "AND conversation_id NOT IN ("
                            "SELECT conversation_id FROM conversation_tags WHERE tag_id = :k)"
                        ),
                        {"k": keeper, "l": loser},
                    )
                    sync_conn.execute(
                        text("DELETE FROM conversation_tags WHERE tag_id = :l"),
                        {"l": loser},
                    )
                sync_conn.execute(
                    text("DELETE FROM tags WHERE id = :l"), {"l": loser}
                )
    except Exception as exc:  # noqa: BLE001 - never block startup
        _log.warning("Tag dedupe migration skipped: %s", exc)


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
        "CREATE INDEX IF NOT EXISTS ix_conversations_owner_updated "
        "ON conversations (owner_id, updated_at)",
        "CREATE INDEX IF NOT EXISTS ix_conv_messages_conv_created "
        "ON conversation_messages (conversation_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_batch_items_job_status "
        "ON batch_items (batch_job_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_conn_audit_conn_created "
        "ON connection_audit_log (connection_id, created_at)",
        # Enforces facet idempotency for deployments whose document_facets table
        # predates the uq_facet_doc_key_value constraint (create_all won't add it
        # to an existing table).
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_facet_doc_key_value "
        "ON document_facets (document_id, key, value_norm)",
        # NOTE: we deliberately do NOT create a unique index on
        # (owner_id, lower(name)) for tags. Replit's publish flow diffs the dev
        # schema against production and replicates any such index *before the
        # app boots* — so _migrate_dedupe_tags never gets to clean pre-existing
        # production duplicates first, and the index creation aborts the
        # publish. Tag uniqueness is instead enforced at the application layer
        # via a transaction-scoped advisory lock around get-or-create (see
        # tags.create_tag and auto_tagger.apply_tags), with _migrate_dedupe_tags
        # merging any legacy duplicates on startup.
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


async def lock_tag_name(db: AsyncSession, owner_id: str, name: str) -> None:
    """Serialize tag get-or-create per (owner, case-insensitive name).

    Without this, two concurrent paths (e.g. Auto-Tag racing a manual create)
    could both pass the case-insensitive existence check and then each insert a
    same-name tag, forking a "folder". A transaction-scoped PostgreSQL advisory
    lock makes the second writer wait for the first to commit, so the
    check-then-insert is effectively atomic. Released automatically at
    commit/rollback. No-op on non-PostgreSQL engines (it is an advisory lock,
    not a constraint) — dev and prod both run PostgreSQL.

    This replaces a DB-level unique index on (owner_id, lower(name)): such an
    index gets replicated to production by Replit's publish-time schema diff
    *before the app boots*, so _migrate_dedupe_tags can't clean pre-existing
    duplicate rows first and the index creation aborts the publish.
    """
    bind = db.bind
    if bind is not None and getattr(bind, "dialect", None) is not None \
            and bind.dialect.name == "postgresql":
        await db.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
            {"k": f"idpkit_tag:{owner_id}:{name.strip().lower()}"},
        )


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
