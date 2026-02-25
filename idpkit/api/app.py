"""IDP Kit FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from idpkit.version import __version__


async def _recover_stale_jobs(session_factory) -> None:
    """Reset any jobs/documents stuck in pending/running from a prior crash."""
    from sqlalchemy import select, update
    from idpkit.db.models import Job, Document
    from datetime import datetime, timezone

    async with session_factory() as db:
        stuck = await db.execute(
            select(Job).where(Job.status.in_(["pending", "running"]))
        )
        stuck_jobs = stuck.scalars().all()
        if not stuck_jobs:
            return

        job_ids = [j.id for j in stuck_jobs]
        doc_ids = [j.document_id for j in stuck_jobs if j.document_id]

        await db.execute(
            update(Job)
            .where(Job.id.in_(job_ids))
            .values(
                status="failed",
                error="Server restarted while job was in progress",
                completed_at=datetime.now(timezone.utc),
            )
        )

        if doc_ids:
            await db.execute(
                update(Document)
                .where(Document.id.in_(doc_ids), Document.status == "indexing")
                .values(status="uploaded")
            )

        await db.commit()

        import logging
        logging.getLogger(__name__).info(
            "Recovered %d stale jobs on startup: %s", len(job_ids), job_ids
        )


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
    from idpkit.db.session import init_db, async_session
    from idpkit.db.seed import seed_default_admin
    from idpkit.plugins import plugin_manager

    await init_db()
    await seed_default_admin(async_session)
    await _migrate_admin_to_superadmin(async_session)
    await _load_rate_limits(async_session)
    await _recover_stale_jobs(async_session)
    plugin_manager.load_entry_points()
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
