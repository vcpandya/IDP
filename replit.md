# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

## Overview
IDP Kit is a Python-based intelligent document processing toolkit with an AI agent. It provides document parsing, indexing, retrieval, processing, and generation capabilities via a FastAPI web application with a Jinja2 template-based frontend.

## Architecture
- **Backend**: Python FastAPI application (`idpkit/api/app.py`)
- **Frontend**: Server-side rendered Jinja2 templates (`idpkit/web/templates/`)
- **Database**: Replit PostgreSQL via SQLAlchemy async + asyncpg (reads `DATABASE_URL` env var; falls back to SQLite if not set)
- **File Storage**: GCS object storage via Replit sidecar when `DEFAULT_OBJECT_STORAGE_BUCKET_ID` and `PRIVATE_OBJECT_DIR` are set; falls back to local filesystem (`./storage`)
- **Entry point**: `run_server.py` — starts uvicorn on port 5000

## Database
All data lives in a single PostgreSQL database (12 tables):
- **Core**: users, documents, jobs, prompts, templates, conversation_messages
- **Knowledge Graph**: entities, entity_mentions, graph_edges (standard relational tables, no graph DB needed)
- **Batch Processing**: processing_templates, batch_jobs, batch_items
- **Connection**: `idpkit/db/session.py` reads `DATABASE_URL`, converts to `postgresql+asyncpg://`, strips sslmode params
- **Schema**: Auto-created via `Base.metadata.create_all()` at startup
- **DateTime handling**: Uses `TZDateTime` custom type decorator to handle timezone-aware datetimes with asyncpg

## Authentication & User Management
- **Admin seeding**: On startup, if no users exist, a default admin is created (username: `admin`, password from `IDP_ADMIN_PASSWORD` env var, default: `admin123`)
- **User approval flow**: New user registrations create accounts with `is_active=0` (pending). Users cannot log in until an admin approves them via the Admin panel.
- **Admin panel**: Available at `/admin/users` (sidebar link visible only to admin users). Admins can approve, deactivate, and delete users.
- **Auth methods**: JWT Bearer token, API key header (`X-API-Key`), or session cookie

## File Storage
- **Abstract interface**: `StorageBackend` in `idpkit/core/storage.py`
- **GCS backend**: `GCSStorageBackend` uses Replit's sidecar API at `127.0.0.1:1106` for signed URL upload/download with local caching
- **Local backend**: `LocalStorageBackend` for development/fallback
- **Selection**: `get_storage()` in `idpkit/api/deps.py` auto-selects based on env vars

## Project Structure
- `idpkit/api/` — FastAPI app factory and API routes (auth, documents, indexing, jobs, retrieval, agent, tools, generation, processing, plugins, graph, batch, admin)
- `idpkit/web/` — Web UI routes and Jinja2 templates
- `idpkit/db/` — Database models, session management, and admin seeding
- `idpkit/engine/` — PageIndex document indexing engine
- `idpkit/parsing/` — Document parsers (PDF, DOCX, HTML, PPTX, etc.)
- `idpkit/processing/` — Document processing pipelines
- `idpkit/retrieval/` — RAG retrieval and search (tree-based, no vector DB)
- `idpkit/agent/` — AI agent with tools
- `idpkit/tools/` — Smart document tools (extract, summarize, compare, etc.)
- `idpkit/graph/` — Knowledge graph builder (entities, edges, cross-doc linking)
- `idpkit/generation/` — Document generation (DOCX, Markdown)
- `idpkit/plugins/` — Plugin system
- `idpkit/core/storage.py` — File storage abstraction (GCS + local)
- `pageindex/` — Standalone PageIndex library

## Environment Variables
- `DATABASE_URL` — PostgreSQL connection string (set by Replit)
- `DEFAULT_OBJECT_STORAGE_BUCKET_ID` — GCS bucket for file storage (set by Replit)
- `PRIVATE_OBJECT_DIR` — Private directory prefix in GCS bucket (set by Replit)
- `IDP_ADMIN_PASSWORD` — Default admin password (default: `admin123`)
- `IDP_SECRET_KEY` — JWT signing key (default: `dev-secret-change-in-production`)

## Running
- Dev: `python run_server.py` (port 5000, host 0.0.0.0)
- Deployment: autoscale target with `python run_server.py`

## Key Dependencies
- FastAPI, Uvicorn, SQLAlchemy (async), asyncpg
- OpenAI, LiteLLM, tiktoken
- PyMuPDF, PyPDF2, python-docx, beautifulsoup4
- Jinja2, Pydantic, httpx
- passlib + bcrypt (auth), python-jose (JWT)
- NetworkX (optional, for graph analytics)
