"""Background runner for smart-metadata (re)extraction jobs.

A job is launched with ``asyncio.create_task`` from the request handler so the
HTTP response returns immediately. The runner owns its own DB session (the
request session is gone once the response is sent) and writes progress to the
``metadata_jobs`` row after every document, so any Gunicorn worker can serve a
poll request.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select, update

from idpkit.core.llm import get_default_client
from idpkit.db.models import Document
from idpkit.db.session import async_session
from idpkit.metadata.extractor import profile_document
from idpkit.metadata.models import MetadataJob

logger = logging.getLogger(__name__)

# Bound how many per-document failures we persist so the JSON column can never
# grow unbounded for a large, mostly-failing batch.
MAX_RECORDED_FAILURES = 50


async def run_extraction_job(job_id: str, owner_id: str, doc_ids: list[str]) -> None:
    """Process *doc_ids* for *owner_id*, updating the ``MetadataJob`` row.

    Per-document failures are isolated (rolled back and counted) so one bad
    document never aborts the batch. Documents lacking both a tree index and a
    description are skipped (nothing to profile).
    """
    try:
        async with async_session() as db:
            llm = get_default_client()

            # Snapshot the work set as plain tuples up front. ``profile_document``
            # commits internally, which expires ORM instances; loading each
            # document fresh by id inside the loop avoids lazy-load-after-commit
            # errors in async.
            rows = (
                await db.execute(
                    select(
                        Document.id, Document.filename,
                        Document.tree_index, Document.description,
                    ).where(
                        Document.id.in_(doc_ids),
                        Document.owner_id == owner_id,
                    )
                )
            ).all()
            work = [
                (r.id, r.filename, bool(r.tree_index) or bool(r.description))
                for r in rows
            ]

            await db.execute(
                update(MetadataJob)
                .where(MetadataJob.id == job_id)
                .values(status="running", total=len(work))
            )
            await db.commit()

            processed = failed = skipped = 0
            failures: list[dict] = []
            for doc_id, filename, has_content in work:
                await db.execute(
                    update(MetadataJob)
                    .where(MetadataJob.id == job_id)
                    .values(current=filename)
                )
                await db.commit()

                if not has_content:
                    skipped += 1
                else:
                    try:
                        doc = await db.get(Document, doc_id)
                        if doc is None or doc.owner_id != owner_id:
                            skipped += 1
                        else:
                            await profile_document(db, llm, doc)
                            processed += 1
                    except Exception as exc:  # noqa: BLE001 - per-doc isolation
                        await db.rollback()
                        failed += 1
                        if len(failures) < MAX_RECORDED_FAILURES:
                            failures.append({
                                "filename": filename or doc_id,
                                "error": str(exc)[:300],
                            })
                        logger.warning(
                            "Metadata job %s: failed on doc %s: %s",
                            job_id, doc_id, exc,
                        )

                await db.execute(
                    update(MetadataJob)
                    .where(MetadataJob.id == job_id)
                    .values(
                        processed=processed,
                        failed=failed,
                        skipped=skipped,
                        failures=json.dumps(failures) if failures else None,
                    )
                )
                await db.commit()

            await db.execute(
                update(MetadataJob)
                .where(MetadataJob.id == job_id)
                .values(status="completed", current=None)
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 - mark the whole job failed
        logger.exception("Metadata job %s crashed: %s", job_id, exc)
        try:
            async with async_session() as db:
                await db.execute(
                    update(MetadataJob)
                    .where(MetadataJob.id == job_id)
                    .values(status="failed", error=str(exc)[:1000], current=None)
                )
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Metadata job %s: could not record failure", job_id)
