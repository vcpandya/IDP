"""IDP Kit FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from idpkit.version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load plugins on startup."""
    from idpkit.db.session import init_db
    from idpkit.plugins import plugin_manager

    await init_db()
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    # Register web UI routes
    from idpkit.web.routes import router as web_router

    app.include_router(web_router)

    return app
