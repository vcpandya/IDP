# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

## Overview
IDP Kit is a Python-based intelligent document processing toolkit with an AI agent. It provides document parsing, indexing, retrieval, processing, and generation capabilities via a FastAPI web application with a Jinja2 template-based frontend.

## Architecture
- **Backend**: Python FastAPI application (`idpkit/api/app.py`)
- **Frontend**: Server-side rendered Jinja2 templates (`idpkit/web/templates/`)
- **Database**: Replit PostgreSQL via SQLAlchemy async + asyncpg (reads `DATABASE_URL` env var; falls back to SQLite if not set)
- **File Storage**: Local filesystem via `LocalStorageBackend` (default: `./storage`)
- **Entry point**: `run_server.py` — starts uvicorn on port 5000

## Database
All data lives in a single PostgreSQL database (12 tables):
- **Core**: users, documents, jobs, prompts, templates, conversation_messages
- **Knowledge Graph**: entities, entity_mentions, graph_edges (standard relational tables, no graph DB needed)
- **Batch Processing**: processing_templates, batch_jobs, batch_items
- **Connection**: `idpkit/db/session.py` reads `DATABASE_URL`, converts to `postgresql+asyncpg://`, strips sslmode params
- **Schema**: Auto-created via `Base.metadata.create_all()` at startup

## Project Structure
- `idpkit/api/` — FastAPI app factory and API routes
- `idpkit/web/` — Web UI routes and Jinja2 templates
- `idpkit/db/` — Database models and session management
- `idpkit/engine/` — PageIndex document indexing engine
- `idpkit/parsing/` — Document parsers (PDF, DOCX, HTML, PPTX, etc.)
- `idpkit/processing/` — Document processing pipelines
- `idpkit/retrieval/` — RAG retrieval and search (tree-based, no vector DB)
- `idpkit/agent/` — AI agent with tools
- `idpkit/tools/` — Smart document tools (extract, summarize, compare, etc.)
- `idpkit/graph/` — Knowledge graph builder (entities, edges, cross-doc linking)
- `idpkit/generation/` — Document generation (DOCX, Markdown)
- `idpkit/plugins/` — Plugin system
- `idpkit/core/storage.py` — File storage abstraction (currently local filesystem)
- `pageindex/` — Standalone PageIndex library

## Running
- Dev: `python run_server.py` (port 5000, host 0.0.0.0)
- Deployment: autoscale target with `python run_server.py`

## Key Dependencies
- FastAPI, Uvicorn, SQLAlchemy (async), asyncpg
- OpenAI, LiteLLM, tiktoken
- PyMuPDF, PyPDF2, python-docx, beautifulsoup4
- Jinja2, Pydantic, httpx
- NetworkX (optional, for graph analytics)
