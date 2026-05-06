# IDP Kit

IDP Kit is a Python-based intelligent document processing toolkit and AI agent for parsing, indexing, retrieving, processing, and generating documents.

## Run & Operate

```bash
# Run the application
uvicorn idpkit.main:app --host 0.0.0.0 --port 8000 --reload

# Production with Gunicorn
gunicorn idpkit.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
```

**Required Environment Variables:**
- `SECRET_KEY`: Stable random value (≥32 chars) for JWT signing and credential encryption.
- `DEPLOYED_DOMAIN`: Canonical production hostname (e.g., `idpkit.example.com`).
- `DATABASE_URL`: PostgreSQL async connection URL.
- `IDP_ADMIN_PASSWORD`: Initial password for the seeded `admin` user.

**Optional Environment Variables:**
- `CORS_EXTRA_ORIGINS`: Comma-separated additional browser origins.
- `OAUTH_REDIRECT_BASE_URL`: Overrides the OAuth callback base URL.
- `EMAIL_API_KEY`: API key for sending e-sign invitations.
- `VISION_MODEL`: Model used for document verification (default `gpt-4o`).
- `ESIGN_BATCH_CONCURRENCY`: Max in-flight envelopes per bulk-send batch (default 3).
- `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_RECYCLE`: PostgreSQL connection pool settings.

## Stack

- **Framework**: FastAPI
- **Runtime**: Python 3.10+
- **ORM**: SQLAlchemy (async)
- **Validation**: Pydantic
- **Build Tool**: Gunicorn, Uvicorn

## Where things live

- `/idpkit`: Core application source code.
    - `/idpkit/api/routes`: API endpoints.
    - `/idpkit/agent`: AI agent implementation and tools.
    - `/idpkit/batch`: Batch processing logic.
    - `/idpkit/connectors`: SaaS connector implementations.
    - `/idpkit/db/models.py`: Database schema definitions.
    - `/idpkit/esign`: E-signature workflow.
        - `templates_lib.py`: Snapshot envelope→template + instantiate template→envelope.
        - `bulk_runner.py`: Async runner for bulk-send batches (bounded concurrency, per-row isolation).
        - `recipient_parsers.py`: CSV / XLSX / pasted-table parsers for bulk recipient lists.
        - `merge.py`: `{{key}}` mail-merge substitution helpers.
    - `/idpkit/verifier`: Document verification engine.
    - `/idpkit/web/templates`: Jinja2 frontend templates.
- `idpkit/core/llm.py`: LLM API key resolution logic.
- `idpkit/core/web_search.py`: Jina AI web search utility.
- `docs/skill-authoring.md`: Guide for creating connector-aware skills.
- `tests/test_esign_e2e.py`: E-signature end-to-end tests.

## Architecture decisions

- **No external vector database**: Retrieval system uses a tree-based approach, loading document content on-demand without a separate vector store.
- **Dynamic Connector Tooling**: Agent dynamically registers tools for *active* user connections only, injecting "Connector Availability" into the system prompt.
- **Secure Credential Handling**: SaaS connector credentials are encrypted with Fernet (key derived from `SECRET_KEY`), decrypted just-in-time, and never logged or exposed to LLM context.
- **Robust E-Signature System**: Full envelope-based e-sign with parallel/sequential signing, bulk-apply fields, HMAC-signed audit certificates, per-token rate limiting (`ESIGN_TOKEN_RATE_*`), one-time public download tokens, geo TTL caching, payload size caps (`ESIGN_MAX_SIG_VALUE_CHARS` / `ESIGN_MAX_TEXT_VALUE_CHARS`), background expiry sweep with PG advisory lock (`ESIGN_EXPIRY_SWEEP_INTERVAL`), envelope delete/extend/reactivate flows, and DocuSign-style typed-signature font picker.
- **E-Sign Templates + Bulk Send (Batch Signing)**: Reusable envelope templates snapshot a PDF + role-based signers + field placements, with declared merge fields (`{{key}}`). "Use template" creates a single envelope with role assignments + merge values; "Bulk Send" instantiates one envelope per row from CSV/XLSX upload or pasted table (5000 row cap, `ESIGN_BATCH_CONCURRENCY`-bounded async runner with per-row failure isolation).
- **Leader-locked daily audit prune**: Ensures `connection_audit_log` pruning runs safely and efficiently across multiple Gunicorn workers using PostgreSQL advisory locks.

## Product

- **Document Processing**: Parsing (PDF, DOCX, HTML, PPTX, YouTube transcripts), indexing, and AI auto-tagging.
- **AI Agent (IDA)**: Equipped with 18 specialized tools for document interaction, knowledge graph querying, report generation, web search, and sandboxed code/browser execution. Supports user-created custom skills.
- **SaaS Connectors**: Pluggable framework with 9 out-of-the-box integrations (Slack, Notion, GitHub, Linear, HubSpot, Dropbox, Jira, AWS S3, Google Workspace), supporting org-wide sharing and user-specific allowlists.
- **Knowledge Graph**: Entity extraction, cross-document linking, visualization, bulk generation, and deep analysis with web enrichment.
- **Batch Processing**: 3-step workflow for processing documents, schema generation from prompts, and formatted DOCX output.
- **E-Signature**: DocuSign-like workflow with sender/signer UIs, field placement, secure signing, audit trails, and bulk-apply field propagation.
- **Document Verifier**: Multimodal AI verification of documents against expected descriptions, supporting various file types and real-time streaming results.
- **Per-User Model Preferences**: Users can set default LLM providers and models, with a clear override chain.

## User preferences

I prefer clear, concise, and structured communication. When making changes, please outline the proposed modifications and their rationale before implementation. For complex features or architectural decisions, provide detailed explanations and consider potential impacts. I favor iterative development and expect regular updates on progress. Do not make changes to files outside the `idpkit/` directory unless explicitly instructed.

## Gotchas

- **Production Startup**: The app refuses to start in production without `SECRET_KEY`, `DEPLOYED_DOMAIN`, `DATABASE_URL`, and `IDP_ADMIN_PASSWORD` environment variables set.
- **MIME Type Validation**: Document uploads are rigorously validated; files with content that contradicts their declared extension will be rejected.
- **LLM API Key Selection**: LLM API keys are auto-selected based on model prefix and loaded strictly from environment variables. Authentication errors fail immediately without retrying.
- **Nix Layer Caching**: Keep Nix packages list minimal to prevent deployment timeouts if layers become uncached.

## Pointers

- **OpenAPI Docs**: `/docs`
- **Skill Authoring Guide**: `docs/skill-authoring.md`
- **E2B SDK Documentation**: _Populate as you build_
- **Jina AI Documentation**: _Populate as you build_