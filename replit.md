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
All data lives in a single PostgreSQL database (15 tables):
- **Core**: users, documents, jobs, prompts, templates, conversations, conversation_messages
- **Knowledge Graph**: entities, entity_mentions, graph_edges (standard relational tables, no graph DB needed)
- **Batch Processing**: processing_templates, batch_jobs, batch_items
- **Tags**: tags, document_tags (many-to-many junction table for document grouping)
- **Document extra columns**: `source_url` (YouTube/external URL), `source_type` (upload/youtube)
- **Connection**: `idpkit/db/session.py` reads `DATABASE_URL`, converts to `postgresql+asyncpg://`, strips sslmode params
- **Schema**: Auto-created via `Base.metadata.create_all()` at startup; includes migration helper to drop/recreate legacy `conversation_messages` table if missing `owner_id` column
- **DateTime handling**: Uses `TZDateTime` custom type decorator to handle timezone-aware datetimes with asyncpg
- **Indexes**: `Document.status` is indexed for fast status-based filtering

## Security
- **JWT signing**: Uses `SESSION_SECRET` env var (required); falls back to ephemeral random key with a logged warning — no hardcoded default
- **CORS**: Reads `ALLOWED_ORIGINS` env var (comma-separated); defaults to wildcard if not set
- **Session cookies**: Login sets `secure=True`, `httponly=True` flags
- **Error responses**: All API exception messages are sanitized — no internal details leaked to clients; full tracebacks logged server-side via `logger.exception()`
- **Rate limiting**: slowapi with admin-exempt key function — defaults 30/min for agent chat, 10/min for batch creation; admins/superadmins are exempt; limits configurable via Admin panel (stored in `system_settings` table); returns 429 on excess
- **File uploads**: SVG uploads blocked (XSS risk); max upload size enforced (50 MB)
- **API keys**: All LLM provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`) read from env only — no `os.environ` mutation at runtime

## Authentication & User Management
- **Roles**: Three-tier hierarchy — `superadmin` > `admin` > `user`
- **Admin seeding**: On startup, if no users exist, a default superadmin is created (username: `admin`, password from `IDP_ADMIN_PASSWORD` env var, default: `admin123`). Existing `admin`-role users named "admin" are auto-migrated to `superadmin`.
- **User approval flow**: New user registrations create accounts with `is_active=0` (pending). Users cannot log in until an admin approves them via the Admin panel.
- **Admin panel**: Available at `/admin/users` (sidebar link visible to admin and superadmin). Features: approve/deactivate users, promote users to admin, demote admins (superadmin only), delete users, configure rate limits.
- **Role hierarchy rules**: Superadmin cannot be deleted/deactivated/demoted by anyone. Only superadmin can delete/deactivate/demote other admins. Any admin can promote regular users to admin.
- **Auth methods**: JWT Bearer token, API key header (`X-API-Key`), or session cookie

## File Storage
- **Abstract interface**: `StorageBackend` in `idpkit/core/storage.py`
- **GCS backend**: `GCSStorageBackend` uses Replit's sidecar API at `127.0.0.1:1106` for signed URL upload/download with local caching
- **Direct upload**: GCS backend supports `get_signed_upload_url()` — clients upload directly to GCS via signed PUT URL, then call `POST /api/documents/{id}/confirm-upload`; local backend falls back to server-proxied multipart upload via `POST /api/documents/{id}/upload-content`
- **Local backend**: `LocalStorageBackend` for development/fallback
- **Selection**: `get_storage()` in `idpkit/api/deps.py` auto-selects based on env vars

## Project Structure
- `idpkit/api/` — FastAPI app factory and API routes (auth, documents, indexing, jobs, retrieval, agent, tools, generation, processing, plugins, graph, batch, admin, settings, tags, youtube)
- `idpkit/api/deps.py` — Shared dependencies: auth, storage, rate limiter
- `idpkit/web/` — Web UI routes and Jinja2 templates
- `idpkit/db/` — Database models, session management, and admin seeding
- `idpkit/engine/` — PageIndex document indexing engine, AI auto-tagger
- `idpkit/parsing/` — Document parsers (PDF, DOCX, HTML, PPTX, YouTube transcripts, etc.)
- `idpkit/youtube/` — YouTube URL resolver, metadata fetcher (Data API v3), playlist/channel enumeration
- `idpkit/processing/` — Document processing pipelines
- `idpkit/retrieval/` — RAG retrieval and search (tree-based, no vector DB)
- `idpkit/agent/` — AI agent with tools
- `idpkit/tools/` — Smart document tools (extract, summarize, compare, etc.)
- `idpkit/graph/` — Knowledge graph builder (entities, edges, cross-doc linking)
- `idpkit/generation/` — Document generation (DOCX, Markdown)
- `idpkit/plugins/` — Plugin system
- `idpkit/core/storage.py` — File storage abstraction (GCS + local)
- `idpkit/core/llm.py` — LLM client (LiteLLM-based, passes API keys via kwargs not env mutation)

## IDA Agent Tools
The IDA agent (`idpkit/agent/agent.py` + `idpkit/agent/tools.py`) has 12 tools:
- **Core**: `search_document`, `list_documents`, `summarize_section`, `extract_data`
- **Knowledge Graph**: `query_graph` (5 operations: find_entity, entity_mentions, related_sections, cross_document_links, document_entities), `find_cross_references`
- **Smart Tools**: `run_smart_tool` (gateway to 13 smart tools)
- **Composition**: `compose_with_context`, `generate_report`, `run_batch`
- **Web (Jina AI)**: `web_search` (via `s.jina.ai`), `fetch_url` (via `r.jina.ai`) — requires `JINA_API_KEY` env var

- `pageindex/` — Standalone PageIndex library

## Environment Variables
- `DATABASE_URL` — PostgreSQL connection string (set by Replit)
- `DEFAULT_OBJECT_STORAGE_BUCKET_ID` — GCS bucket for file storage (set by Replit)
- `PRIVATE_OBJECT_DIR` — Private directory prefix in GCS bucket (set by Replit)
- `IDP_ADMIN_PASSWORD` — Default admin password (default: `admin123`)
- `SESSION_SECRET` — JWT signing key (required for production; ephemeral random fallback in dev)
- `ALLOWED_ORIGINS` — Comma-separated CORS allowed origins (optional)
- `JINA_API_KEY` — Jina AI API key for web search (`s.jina.ai`) and URL reader (`r.jina.ai`) tools in IDA agent (optional)
- `YOUTUBE_API_KEY` — YouTube Data API v3 key for video metadata, playlist/channel resolution (required for YouTube ingestion)

## Running
- Dev: `python run_server.py` (port 5000, host 0.0.0.0)
- Deployment: autoscale target with `python run_server.py`

## Key Dependencies
- FastAPI, Uvicorn, SQLAlchemy (async), asyncpg
- OpenAI, LiteLLM, tiktoken
- PyMuPDF, PyPDF2, python-docx, beautifulsoup4
- Jinja2, Pydantic, httpx
- passlib + bcrypt (auth), python-jose (JWT)
- slowapi (rate limiting)
- NetworkX (optional, for graph analytics)
- youtube-transcript-api (YouTube transcript extraction)
- google-api-python-client (YouTube Data API v3)

## Retrieval / Search Pipeline
- **Tree index format**: `{"doc_name": ..., "doc_description": ..., "structure": [...]}` — the `structure` key holds the actual hierarchical node list
- **Node fields**: `title`, `start_index`, `end_index`, `node_id`, `summary`, optional `text`, optional `nodes` (children)
- **Search flow**: `_flatten_tree()` unwraps the `structure` key, flattens all nodes recursively, LLM ranks by title+summary, then actual PDF text is loaded on-demand for matched page ranges
- **Text loading**: Nodes typically don't store inline text (config `if_add_node_text: "no"`). The search tool loads PDF pages from storage using `get_page_tokens()` for matched sections
- **Config defaults** (`idpkit/engine/config.yaml`): model=gpt-4o, toc_check_page_num=20, max_page_num_each_node=10, if_add_node_id=yes, if_add_node_summary=yes, if_add_node_text=no

## UI/UX Features
- **Favicon**: SVG favicon at `idpkit/web/static/favicon.svg` — document with search icon in indigo gradient
- **Agent Chat**: Tool call messages render as collapsible accordions; source citations open in popup modal with text preview. Supports `?prompt=` URL param to pre-populate the input field (used by dashboard tiles).
- **Dashboard tiles**: "Knowledge Graph" links to `/graph`; "Cross-Document Search", "Entity Discovery", and "Report Generation" tiles link to `/chat?prompt=...` with contextual example prompts pre-filled in the chat input.
- **Knowledge Graph page** (`/graph`): Dedicated explorer with entity search, type filtering, D3 force-directed graph visualization, entity detail panel with mentions and relationships, and "Ask IDA" link. Supports multi-document selection (add/remove individual docs) and tag-based filtering (select a tag group to load all its indexed documents). Multi-doc graph uses `GET /api/graph/visualization?doc_ids=...&tag_id=...` endpoint backed by `get_multi_doc_visualization_data()`. Entity types loaded dynamically from `/api/graph/entity-types`; icons auto-resolve for unknown types via keyword matching. Added to sidebar nav.
- **YouTube Ingestion**: Knowledge Base and Upload pages have a YouTube URL input panel. Accepts video, playlist, or channel URLs. Transcripts are extracted with temporal segmentation (2-min windows with timestamps as structural elements like page numbers). Videos become Document records with `format="youtube"`. Playlists/channels resolve to multiple documents (capped at 100 videos). Supports auto-tag and auto-index options. Progress polling shows real-time status.
- **AI Auto-Tagging**: Auto-tag button on document rows in Knowledge Base. Uses LLM to analyze document content (filename, metadata, description, tree structure) and suggest 1-3 tags. Prefers existing user tags when they match. Can auto-apply suggestions. Also available during YouTube ingestion via checkbox. API: `POST /api/documents/{id}/auto-tag`.
- **Cumulative entity type bank**: Entity extraction no longer enforces a hardcoded type whitelist. The LLM prompt includes all existing entity types from the DB (merged with defaults) and allows creating new UPPER_SNAKE_CASE types when none fit. Types validated by regex format only. This builds a growing corpus across documents.
- **Batch Processing**: Redesigned with 3-step flow (Instructions → Select Documents to Process → Settings). Reference documents attach to the prompt (Step 1) as AI context. Two-pass schema generation: when no template is selected, Pass 1 generates a JSON schema from the prompt + reference docs, Pass 2 uses that schema as structured output constraint for all target documents.
- **Document Viewer**: Tree structure with D3 visualization, outline view, and JSON view; `$watch` on viewMode for dynamic D3 rendering
- **Settings**: LLM model lists fetched dynamically from provider APIs (OpenAI, Anthropic, Google, OpenRouter, Ollama) with curated fallbacks. Model filter search icon positioned on the right to avoid text overlap.

## Performance
- **N+1 fix**: `list_conversations` uses a single LEFT JOIN + GROUP BY query instead of N+1 message count queries
- **Query bounds**: All list endpoints have `.limit()` caps (50-200 depending on endpoint)
- **Direct uploads**: Large files bypass the server entirely when GCS is active (signed PUT URL)
