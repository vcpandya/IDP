# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

## Overview
IDP Kit is a Python-based intelligent document processing toolkit with an AI agent. It provides document parsing, indexing, retrieval, processing, and generation capabilities via a FastAPI web application with a Jinja2 template-based frontend.

## Architecture
- **Backend**: Python FastAPI application (`idpkit/api/app.py`)
- **Frontend**: Server-side rendered Jinja2 templates (`idpkit/web/templates/`)
- **Database**: SQLite via SQLAlchemy async (default: `idpkit.db`)
- **Entry point**: `run_server.py` — starts uvicorn on port 5000

## Project Structure
- `idpkit/api/` — FastAPI app factory and API routes
- `idpkit/web/` — Web UI routes and Jinja2 templates
- `idpkit/db/` — Database models and session management
- `idpkit/engine/` — PageIndex document indexing engine
- `idpkit/parsing/` — Document parsers (PDF, DOCX, HTML, PPTX, etc.)
- `idpkit/processing/` — Document processing pipelines
- `idpkit/retrieval/` — RAG retrieval and search
- `idpkit/agent/` — AI agent with tools
- `idpkit/tools/` — Smart document tools (extract, summarize, compare, etc.)
- `idpkit/graph/` — Knowledge graph builder
- `idpkit/generation/` — Document generation (DOCX, Markdown)
- `idpkit/plugins/` — Plugin system
- `pageindex/` — Standalone PageIndex library

## Running
- Dev: `python run_server.py` (port 5000, host 0.0.0.0)
- Deployment: autoscale target with `python run_server.py`

## Key Dependencies
- FastAPI, Uvicorn, SQLAlchemy (async), aiosqlite
- OpenAI, LiteLLM, tiktoken
- PyMuPDF, PyPDF2, python-docx, beautifulsoup4
- Jinja2, Pydantic, httpx
