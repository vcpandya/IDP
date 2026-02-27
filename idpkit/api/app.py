"""IDP Kit FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from idpkit.version import __version__


async def _recover_stale_jobs(session_factory) -> list:
    """Find jobs stuck in pending/running from a prior crash and prepare them for auto-resume.

    Uses SELECT ... FOR UPDATE SKIP LOCKED to prevent multiple Gunicorn workers
    from claiming the same job. Returns a list of dicts with the info needed to
    re-queue each indexing job. Non-indexing jobs are simply marked as failed.
    """
    from sqlalchemy import select, update
    from idpkit.db.models import Job, Document
    from datetime import datetime, timezone
    import logging

    _log = logging.getLogger(__name__)
    resumable = []

    async with session_factory() as db:
        stuck = await db.execute(
            select(Job)
            .where(Job.status.in_(["pending", "running"]))
            .with_for_update(skip_locked=True)
        )
        stuck_jobs = stuck.scalars().all()
        if not stuck_jobs:
            return []

        for job in stuck_jobs:
            if job.job_type == "index" and job.document_id:
                doc_result = await db.execute(
                    select(Document).where(Document.id == job.document_id)
                )
                doc = doc_result.scalar_one_or_none()
                if doc and doc.file_path and doc.status != "indexed":
                    restart_log = {
                        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        "level": "WARN",
                        "msg": "Resuming after server restart...",
                    }
                    job.status = "resuming"
                    job.progress = 0
                    job.stage = "Resuming after restart"
                    job.error = None
                    job.result = None
                    job.completed_at = None
                    existing_logs = job.logs or []
                    job.logs = existing_logs + [restart_log]
                    db.add(job)

                    if doc.status == "indexing":
                        doc.status = "uploaded"
                        db.add(doc)

                    resumable.append({
                        "job_id": job.id,
                        "doc_id": job.document_id,
                        "user_id": doc.owner_id,
                        "storage_path": doc.file_path,
                        "params": job.params or {},
                    })
                    _log.info("Will auto-resume indexing job %s for doc %s", job.id, job.document_id)
                    continue

            job.status = "failed"
            job.error = "Server restarted while job was in progress"
            job.completed_at = datetime.now(timezone.utc)
            db.add(job)

            if job.document_id:
                doc_result = await db.execute(
                    select(Document).where(
                        Document.id == job.document_id,
                        Document.status == "indexing",
                    )
                )
                stale_doc = doc_result.scalar_one_or_none()
                if stale_doc:
                    stale_doc.status = "uploaded"
                    db.add(stale_doc)

        await db.commit()

        total = len(stuck_jobs)
        _log.info(
            "Recovered %d stale jobs on startup (%d will auto-resume)",
            total, len(resumable),
        )

    return resumable


async def _migrate_admin_to_superadmin(session_factory) -> None:
    """Promote the original seeded admin to superadmin if still 'admin' role."""
    from sqlalchemy import select, update
    from idpkit.db.models import User, UserRole
    import logging

    async with session_factory() as db:
        result = await db.execute(
            select(User).where(User.username == "admin", User.role == UserRole.ADMIN.value)
        )
        original = result.scalar_one_or_none()
        if original:
            original.role = UserRole.SUPERADMIN.value
            db.add(original)
            await db.commit()
            logging.getLogger(__name__).info(
                "Migrated original admin user to superadmin role."
            )


async def _load_rate_limits(session_factory) -> None:
    """Load rate limit settings from DB into cache."""
    from sqlalchemy import select
    from idpkit.db.models import SystemSetting
    from idpkit.api.deps import update_rate_limit_cache
    import json
    import logging

    try:
        async with session_factory() as db:
            result = await db.execute(
                select(SystemSetting).where(SystemSetting.key == "rate_limits")
            )
            setting = result.scalar_one_or_none()
            if setting:
                limits = json.loads(setting.value)
                update_rate_limit_cache(limits)
                logging.getLogger(__name__).info("Loaded rate limits from DB: %s", limits)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load rate limits: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load plugins on startup."""
    import asyncio
    from idpkit.db.session import init_db, async_session
    from idpkit.db.seed import seed_default_admin
    from idpkit.plugins import plugin_manager

    await init_db()
    await seed_default_admin(async_session)
    await _migrate_admin_to_superadmin(async_session)
    await _load_rate_limits(async_session)
    resumable = await _recover_stale_jobs(async_session)
    plugin_manager.load_entry_points()

    if resumable:
        import logging
        from idpkit.api.routes.indexing import _run_indexing_task
        _log = logging.getLogger(__name__)
        for info in resumable:
            _log.info("Auto-resuming indexing job %s", info["job_id"])
            asyncio.create_task(_run_indexing_task(
                job_id=info["job_id"],
                doc_id=info["doc_id"],
                user_id=info["user_id"],
                storage_path=info["storage_path"],
                params=info["params"],
            ))

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="IDP Kit",
        description="Intelligent Document Processing Toolkit & AI Agent",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    import os
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from idpkit.api.deps import limiter, decode_token
    app.state.limiter = limiter

    @app.middleware("http")
    async def inject_user_role(request: Request, call_next):
        role = None
        auth = request.headers.get("authorization", "")
        token = None
        if auth.startswith("Bearer "):
            token = auth[7:]
        if not token:
            token = request.cookies.get("session_token")
        if token:
            user_id = decode_token(token)
            if user_id:
                from idpkit.db.session import async_session as _sf
                from idpkit.db.models import User as _U
                from sqlalchemy import select as _sel
                async with _sf() as _s:
                    _r = await _s.execute(_sel(_U.role).where(_U.id == user_id))
                    row = _r.scalar_one_or_none()
                    if row:
                        role = row
        request.state._user_role = role
        return await call_next(request)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please slow down."},
        )

    # Mount static files
    static_dir = Path(__file__).parent.parent / "web" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": __version__}

    # API info
    @app.get("/api")
    async def api_info():
        return {
            "name": "IDP Kit API",
            "version": __version__,
            "description": "Intelligent Document Processing Toolkit & AI Agent",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "api": "/api",
            },
        }

    # Register API routes
    from idpkit.api.routes import (
        auth_router,
        documents_router,
        indexing_router,
        jobs_router,
        retrieval_router,
        agent_router,
        tools_router,
        generation_router,
        processing_router,
        plugins_router,
        graph_router,
        batch_router,
        admin_router,
        settings_router,
        tags_router,
        youtube_router,
        skills_router,
        verifier_router,
    )

    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(indexing_router)
    app.include_router(jobs_router)
    app.include_router(retrieval_router)
    app.include_router(agent_router)
    app.include_router(tools_router)
    app.include_router(generation_router)
    app.include_router(processing_router)
    app.include_router(plugins_router)
    app.include_router(graph_router)
    app.include_router(batch_router)
    app.include_router(admin_router)
    app.include_router(settings_router)
    app.include_router(tags_router)
    app.include_router(youtube_router)
    app.include_router(skills_router)
    app.include_router(verifier_router)

    # Register web UI routes
    from idpkit.web.routes import router as web_router

    app.include_router(web_router)

    return app
