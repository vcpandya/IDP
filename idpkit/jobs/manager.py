"""Job manager for IDP Kit — runs indexing jobs with DB bookkeeping."""

import logging
import traceback
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def run_indexing_job(
    job_id: str,
    doc_id: str,
    file_path: str,
    format: str,
    options: dict | None = None,
    db_url: str | None = None,
) -> dict:
    """Execute an indexing job: build the tree index and persist it.

    This function is designed to be called as a background task (e.g. via
    ``asyncio.create_task``, an ``anyio`` task group, or a Celery/ARQ worker).
    It creates its **own** database session so it never shares a session with
    the originating HTTP request.

    Parameters
    ----------
    job_id:
        Primary key of the :class:`Job` row.
    doc_id:
        Primary key of the :class:`Document` row to update.
    file_path:
        Path to the uploaded file on disk.
    format:
        File extension including the leading dot (e.g. ``".pdf"``).
    options:
        Optional dict of engine options forwarded to the indexer.
    db_url:
        Optional database URL.  When *None* the default
        ``async_session`` from ``idpkit.db.session`` is used.

    Returns
    -------
    dict
        The tree-index result dict on success.

    Raises
    ------
    Exception
        Re-raised after the job/document status has been set to *failed*.
    """
    if options is None:
        options = {}

    # ------------------------------------------------------------------
    # Obtain a session factory
    # ------------------------------------------------------------------
    session_factory = await _get_session_factory(db_url)

    # ------------------------------------------------------------------
    # 1. Mark job as running
    # ------------------------------------------------------------------
    async with session_factory() as session:
        try:
            await _update_job_status(session, job_id, status="running", progress=0)
            await _update_document_status(session, doc_id, status="indexing")
            await session.commit()
        except Exception:
            await session.rollback()
            raise

    # ------------------------------------------------------------------
    # 2-3. Run the indexer
    # ------------------------------------------------------------------
    try:
        from idpkit.indexing import get_indexer

        indexer = get_indexer(format)
        logger.info(
            "Job %s: starting %s indexing for document %s (%s)",
            job_id,
            type(indexer).__name__,
            doc_id,
            file_path,
        )

        result = await indexer.build_index(file_path, **options)

    except Exception as exc:
        # ------------------------------------------------------------------
        # 7. On error — mark both job and document as failed
        # ------------------------------------------------------------------
        error_message = f"{type(exc).__name__}: {exc}"
        error_tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, error_message, error_tb)

        async with session_factory() as session:
            try:
                await _update_job_status(
                    session,
                    job_id,
                    status="failed",
                    progress=0,
                    error=error_message,
                    completed_at=_utcnow(),
                )
                await _update_document_status(session, doc_id, status="failed")
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception("Job %s: failed to persist failure status", job_id)
        raise

    # ------------------------------------------------------------------
    # 4-6. Persist the result
    # ------------------------------------------------------------------
    async with session_factory() as session:
        try:
            await _save_index_result(session, doc_id, result)
            await _update_document_status(session, doc_id, status="indexed")
            await _update_job_status(
                session,
                job_id,
                status="completed",
                progress=100,
                result=result,
                completed_at=_utcnow(),
            )
            await session.commit()
            logger.info("Job %s: completed successfully", job_id)
        except Exception:
            await session.rollback()
            logger.exception("Job %s: failed to persist success result", job_id)
            raise

    # ------------------------------------------------------------------
    # 7. Build knowledge graph (non-fatal)
    # ------------------------------------------------------------------
    try:
        await _build_graph_for_document(doc_id, result, session_factory)
    except Exception:
        logger.warning(
            "Job %s: graph building failed (non-fatal), indexing still succeeded",
            job_id,
            exc_info=True,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _get_session_factory(db_url: str | None):
    """Return an async session factory.

    When *db_url* is ``None`` we reuse the application-wide session factory
    from ``idpkit.db.session``.  Otherwise we spin up a one-off engine (useful
    for workers running in a separate process).
    """
    if db_url is None:
        from idpkit.db.session import async_session
        return async_session

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    _engine = create_async_engine(db_url, echo=False)
    return async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def _update_job_status(
    session,
    job_id: str,
    *,
    status: str,
    progress: int | None = None,
    result: dict | None = None,
    error: str | None = None,
    completed_at: datetime | None = None,
) -> None:
    """Update fields on an existing :class:`Job` row."""
    from sqlalchemy import select
    from idpkit.db.models import Job

    stmt = select(Job).where(Job.id == job_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        logger.warning("Job %s not found in DB — cannot update status", job_id)
        return

    row.status = status
    if progress is not None:
        row.progress = progress
    if result is not None:
        row.result = result
    if error is not None:
        row.error = error
    if completed_at is not None:
        row.completed_at = completed_at


async def _update_document_status(session, doc_id: str, *, status: str) -> None:
    """Update the status field on a :class:`Document` row."""
    from sqlalchemy import select
    from idpkit.db.models import Document

    stmt = select(Document).where(Document.id == doc_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        logger.warning("Document %s not found in DB — cannot update status", doc_id)
        return

    row.status = status
    row.updated_at = _utcnow()


async def _save_index_result(session, doc_id: str, result: dict) -> None:
    """Persist the tree index and optional description on a Document."""
    from sqlalchemy import select
    from idpkit.db.models import Document

    stmt = select(Document).where(Document.id == doc_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        logger.warning("Document %s not found in DB — cannot save index", doc_id)
        return

    row.tree_index = result.get("structure")
    if "doc_description" in result:
        row.description = result["doc_description"]
    row.updated_at = _utcnow()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _build_graph_for_document(
    doc_id: str,
    index_result: dict,
    session_factory,
) -> None:
    """Build knowledge graph entities and edges for an indexed document.

    This is called after a successful indexing job. Failures here are
    non-fatal — the indexing result is already persisted.
    """
    from idpkit.core.llm import get_default_client
    from idpkit.graph.builder import build_document_graph
    from idpkit.graph.linker import link_entities_across_documents

    llm = get_default_client()
    tree_index = {"structure": index_result.get("structure", [])}

    if not tree_index["structure"]:
        logger.debug("Skipping graph building for doc %s: empty structure", doc_id)
        return

    async with session_factory() as session:
        try:
            build_result = await build_document_graph(doc_id, tree_index, llm, session)
            logger.info(
                "Graph built for doc %s: %d entities, %d edges",
                doc_id,
                build_result.get("entity_count", 0),
                build_result.get("edge_count", 0),
            )

            link_result = await link_entities_across_documents(doc_id, session, llm)
            logger.info(
                "Cross-doc links for doc %s: %d exact, %d fuzzy",
                doc_id,
                link_result.get("exact_links", 0),
                link_result.get("fuzzy_links", 0),
            )
        except Exception:
            await session.rollback()
            raise
