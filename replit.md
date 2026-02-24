# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

## Overview
IDP Kit is a Python-based intelligent document processing toolkit designed for document parsing, indexing, retrieval, processing, and generation. It features an AI agent and is delivered as a FastAPI web application with a Jinja2 template-based frontend. The project aims to provide comprehensive tools for managing and extracting insights from various document types, supporting an end-to-end intelligent document processing workflow.

## User Preferences
I prefer clear, concise, and structured communication. When making changes, please outline the proposed modifications and their rationale before implementation. For complex features or architectural decisions, provide detailed explanations and consider potential impacts. I favor iterative development and expect regular updates on progress. Do not make changes to files outside the `idpkit/` directory unless explicitly instructed.

## System Architecture
The application follows a client-server architecture:
-   **Backend**: Python FastAPI application.
-   **Frontend**: Server-Side Rendered (SSR) Jinja2 templates for the web UI.
-   **Database**: PostgreSQL is the primary database, managed with SQLAlchemy (async). It includes tables for core data, knowledge graphs, batch processing, and tags. The schema is auto-created on startup.
-   **File Storage**: An abstract `StorageBackend` interface supports Google Cloud Storage (GCS) via Replit sidecar for production and local filesystem for development. GCS supports direct client uploads via signed URLs.
-   **Authentication & Authorization**: Features a three-tier role hierarchy (`superadmin`, `admin`, `user`), JWT-based authentication, API key support, and a user approval workflow for new registrations. An admin panel facilitates user management and rate limit configuration.
-   **Security**: Employs JWT signing, CORS configuration, sanitized API error responses, rate limiting, and restricts file uploads to prevent XSS. LLM API keys are strictly loaded from environment variables.
-   **Document Processing**: Includes parsers for various document types (PDF, DOCX, HTML, PPTX, YouTube transcripts), an indexing engine (`PageIndex`), and an AI auto-tagger.
-   **Retrieval**: Utilizes a tree-based retrieval system without an external vector database, where search results load document content on-demand.
-   **AI Agent**: The IDA agent (`idpkit/agent/agent.py`) is equipped with 12 specialized tools for document interaction, knowledge graph querying, smart tool execution, report generation, and web search capabilities (Jina AI).
-   **Knowledge Graph**: A dedicated module for building and querying knowledge graphs, including entity extraction, cross-document linking, visualization, and bulk graph generation. The `POST /api/graph/build-bulk` endpoint accepts `{ "document_ids": [...] }` and sequentially builds graphs for multiple documents, auto-indexing unindexed docs first, skipping those already built. Both the Knowledge Graph page and Knowledge Base tag detail view offer "Index & Build Graphs" buttons with real-time progress feedback.
-   **Batch Processing**: A redesigned 3-step workflow for processing documents in batches, supporting schema generation from prompts.
-   **UI/UX**: Features an SVG favicon, interactive agent chat with collapsible tool calls and source citations, dashboard tiles for quick access to features, a dedicated Knowledge Graph explorer with D3 visualization and export options, and YouTube ingestion with a video preview/selection modal. Knowledge Base filtering, bulk actions, and AI auto-tagging are also integrated. The document viewer supports tree, outline, and JSON views. LLM model lists are dynamically fetched.
-   **Performance**: N+1 query fixes and query limits are implemented in list endpoints.

## External Dependencies
-   **Database Drivers**: `asyncpg`
-   **Web Framework**: `FastAPI`, `Uvicorn`
-   **ORM**: `SQLAlchemy`
-   **Templating**: `Jinja2`
-   **LLM Integration**: `OpenAI`, `LiteLLM`, `tiktoken` (for tokenization)
-   **Document Parsing**: `PyMuPDF`, `PyPDF2`, `python-docx`, `beautifulsoup4`
-   **Authentication**: `passlib`, `bcrypt`, `python-jose`
-   **Rate Limiting**: `slowapi`
-   **HTTP Client**: `httpx`
-   **Data Validation**: `Pydantic`
-   **Graph Analytics**: `NetworkX` (optional)
-   **YouTube Integration**: `youtube-transcript-api`, `google-api-python-client` (YouTube Data API v3)
-   **Third-Party APIs**:
    -   `Jina AI` (for `web_search` and `fetch_url` agent tools)
    -   `Supadata API` (fallback for YouTube transcript extraction)
    -   `Webshare Proxy` (for proxied YouTube transcript fetching)