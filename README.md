# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

**IDP Kit** is a full-stack Python toolkit for intelligent document processing. It combines tree-based document indexing, a knowledge graph, an AI agent with tool-calling, 13 smart document tools, and a web UI — all in one package.

Upload PDFs, DOCX, HTML, spreadsheets, presentations, or images. IDP Kit indexes them into a hierarchical tree structure, extracts a knowledge graph of entities and relationships, and lets you query everything through an AI agent or REST API.

## Features

| Category | What you get |
|----------|-------------|
| **Document Indexing** | Hierarchical tree index — no vector DB, no chunking, pure LLM reasoning |
| **Knowledge Graph** | Automatic entity extraction, cross-document linking, relationship discovery |
| **AI Agent (IDA)** | Tool-calling agent (6 tools) — search, summarize, extract, graph queries, cross-references |
| **Smart Tools** | 13 tools: Summary, Classify, Extract, Compare, Q&A, Split, Redaction, Anonymize, Fill, Rewrite, Translate, Merge, Audit |
| **Batch Processing** | Two-pass AI-powered schema generation, concurrent multi-document processing with reference docs |
| **Multi-Format** | PDF, DOCX, HTML, XLSX/CSV, PPTX, Images (OCR) |
| **Multi-Provider LLM** | OpenAI, Anthropic, Gemini, Ollama, OpenRouter — all via [LiteLLM](https://github.com/BerriAI/litellm) |
| **Direct-to-Storage Upload** | Signed URL uploads bypass the server for large files (GCS) |
| **Admin Panel** | Three-tier role hierarchy, user approval, configurable rate limits |
| **Web UI** | FastAPI + Jinja2 + Alpine.js |
| **REST API** | 60+ endpoints across 13 routers |
| **Plugin System** | Entry-point + directory-based plugin discovery |
| **Database** | SQLAlchemy 2.0 async — SQLite (dev) / PostgreSQL (prod) |

---

## Security & Production Hardening

IDP Kit is hardened for production deployment:

| Feature | Details |
|---------|---------|
| **JWT Authentication** | Signed with `SESSION_SECRET` env var — no hardcoded fallbacks. Ephemeral random key in dev with logged warning. |
| **CORS** | Configurable via `ALLOWED_ORIGINS` env var (comma-separated). Defaults to wildcard if not set. |
| **Session Cookies** | `secure=True`, `httponly=True` flags on login cookies |
| **Rate Limiting** | Admin-configurable limits (default: 30/min chat, 10/min batch). Admins and superadmins are exempt. Returns 429 on excess. |
| **Error Sanitization** | No internal details leaked in HTTP error responses. Full tracebacks logged server-side only. |
| **File Upload Safety** | SVG uploads blocked (XSS risk). 50 MB max file size enforced on all upload paths. |
| **Direct Uploads** | Large files upload directly to GCS via signed PUT URLs — server never receives the file body. |
| **API Key Safety** | All LLM provider keys read from environment only — no `os.environ` mutation at runtime. |
| **Query Bounds** | All list endpoints have `.limit()` caps to prevent unbounded queries. |

---

## User Management & Roles

IDP Kit uses a three-tier role hierarchy:

| Role | Capabilities |
|------|-------------|
| **Superadmin** | Full control. Can promote/demote/delete/deactivate any user. Cannot be deleted or demoted. |
| **Admin** | Can approve/deactivate regular users, promote users to admin, configure rate limits. Cannot modify other admins. |
| **User** | Standard access to documents, tools, agent, and batch processing. |

- On first startup, a default **superadmin** is created (username: `admin`, password from `IDP_ADMIN_PASSWORD` env var, default: `admin123`).
- New user registrations are **pending** by default — an admin must approve them before they can log in.
- The Admin panel (`/admin/users`) provides user management, role promotion/demotion, and rate limit configuration.

---

## Use Cases

### Bulk Document Processing

Process hundreds of documents at scale through the REST API or processing pipeline.

- **Batch Upload & Index** — Upload entire document libraries via the API. Each document is automatically parsed, indexed into a tree structure, and added to the knowledge graph.
- **Two-Pass Schema Generation** — For batch processing, Pass 1 analyzes your plain-text instructions to generate a custom JSON schema. Pass 2 uses that schema as a structured output constraint for processing all target documents.
- **Reference Documents** — Attach reference documents to your batch prompt as AI context, so the model uses them when processing target documents.
- **Multi-Format Ingestion** — Process mixed collections containing PDFs, Word documents, HTML pages, spreadsheets, presentations, and scanned images in a single pipeline.
- **Background Job Tracking** — All indexing runs as background jobs with status monitoring, so you can queue up large batches and track progress via the Jobs API.

### Document Analysis & Intelligence

Extract insights from documents without reading them cover to cover.

- **Smart Summary** — Generate hierarchical summaries at configurable lengths (brief, standard, detailed) with custom tone and target audience. Summarize an entire 200-page report or drill into specific sections.
- **Smart Classify** — Automatically categorize documents by type (contract, invoice, report, memo), topics, sentiment, or urgency level — with confidence scores.
- **Smart Extract** — Pull structured data out of unstructured documents: tables, key-value pairs, named entities, financial figures, or any custom schema you define.
- **Smart Q&A** — Auto-generate question-answer pairs from documents for training datasets, quizzes, FAQs, or knowledge base seeding.
- **Smart Audit** — Audit documents against compliance standards, style guides, or completeness checklists. Get a scored report with violations and recommendations.

### Document Comparison & Cross-Referencing

Compare documents and discover connections across your entire library.

- **Smart Compare** — Compare two documents structurally, semantically, or for regulatory alignment. Identify what changed between versions, what's missing, and what conflicts.
- **Cross-Document Search** — Query across all indexed documents simultaneously. The AI agent finds and cites relevant sections from multiple sources.
- **Knowledge Graph Links** — Automatically discover which documents share entities (people, organizations, regulations, concepts). Ask "Which other documents mention Algorithm X?" and get answers instantly.
- **Entity Tracking** — Track people, organizations, regulations, products, and other entities across your entire document library. See every mention, every relationship, every cross-reference.

### Data Privacy & Compliance

Handle sensitive documents with built-in privacy tools.

- **Smart Redaction** — Automatically detect and redact PII (names, emails, phone numbers, SSNs, addresses) with a full audit trail. Choose auto-detection or selective redaction by entity type.
- **Smart Anonymize** — Replace real identities with consistent, realistic pseudonyms. "John Smith" becomes "David Brown" everywhere in the document — consistently, so relationships are preserved.
- **Compliance Auditing** — Run automated compliance checks against regulatory standards. Get a scored report identifying gaps and violations.
- **Audit Trail** — Every redaction and anonymization operation is logged for regulatory compliance.

### Content Transformation & Rewriting

Transform documents to meet different needs.

- **Smart Rewrite** — Improve clarity, simplify language, formalize tone, or enrich context. Adjust reading level for different audiences (technical → executive summary, legal → plain language).
- **Smart Translate** — Translate documents while preserving structure, formatting, and domain terminology. Supply a glossary for industry-specific terms.
- **Format Conversion** — Convert between Markdown, DOCX, and other formats while preserving content and structure.
- **Document Generation** — Generate polished DOCX or Markdown documents from templates, tree indices, or AI-generated content.

### Document Assembly & Automation

Combine, split, and automate document workflows.

- **Smart Merge** — Combine multiple documents by concatenation, deduplication, or intelligent synthesis. Merge related reports into a single unified document.
- **Smart Split** — Break large documents into meaningful chunks by heading structure, size, or semantic boundaries — preserving hierarchy and context.
- **Smart Fill** — Auto-populate templates by extracting data from source documents and mapping it to template fields. Fill contracts, forms, and reports from source data.
- **Processing Pipelines** — Chain multiple operations into automated workflows. Extract → Redact → Translate → Generate DOCX in a single pipeline.

---

### Industry Use Cases

<details>
<summary><b>Legal & Contract Management</b></summary>

- Upload and index large contract libraries across clients
- Extract key clauses, dates, obligations, and parties from contracts
- Compare contract versions to identify changes and conflicts
- Redact sensitive client information before sharing with third parties
- Cross-reference regulatory citations across all legal documents
- Auto-generate contract summaries for case review
- Audit contracts against compliance checklists
- Track specific clauses and terms across the entire document library via the knowledge graph

</details>

<details>
<summary><b>Financial Services & Audit</b></summary>

- Process quarterly and annual reports, 10-K filings, and financial statements
- Extract financial figures, metrics, and KPIs from unstructured reports
- Compare financial reports across periods to identify trends and anomalies
- Cross-reference regulatory requirements across compliance documents
- Generate audit reports with compliance scores
- Anonymize client data in financial documents before external review
- Track entities (companies, regulations, financial instruments) across all filings

</details>

<details>
<summary><b>Human Resources</b></summary>

- Batch-process resumes and extract structured candidate profiles (skills, experience, education)
- Classify incoming documents (applications, policies, complaints, evaluations)
- Anonymize employee information in documents shared for analytics or legal review
- Compare policy versions across updates
- Auto-fill offer letters, contracts, and onboarding forms from HR databases
- Generate Q&A pairs from training materials for onboarding quizzes
- Translate HR policies for international offices

</details>

<details>
<summary><b>Research & Knowledge Management</b></summary>

- Index research papers, whitepapers, and technical documentation
- Build a knowledge graph connecting researchers, methodologies, findings, and citations across papers
- Ask the AI agent: "What papers discuss technique X and what were their results?"
- Cross-reference findings across studies to identify consensus and contradictions
- Summarize lengthy research papers at different detail levels
- Generate literature review sections from multiple indexed papers
- Extract and compare methodologies across studies

</details>

<details>
<summary><b>Healthcare & Life Sciences</b></summary>

- Process clinical trial reports, regulatory submissions, and medical literature
- Extract drug names, dosages, adverse events, and outcomes from clinical documents
- Redact patient identifying information (PHI) for HIPAA compliance
- Compare treatment protocols across studies
- Track regulatory citations and compliance requirements across submissions
- Translate medical documents while preserving terminology accuracy with custom glossaries

</details>

<details>
<summary><b>Government & Public Sector</b></summary>

- Batch-process FOIA requests with automatic PII redaction
- Index policy documents, regulations, and legislative texts
- Cross-reference regulations across agencies and jurisdictions
- Classify incoming correspondence by topic, urgency, and department
- Translate public documents for multilingual constituents
- Audit documents against accessibility and compliance standards
- Track entities (programs, agencies, regulations) across government documentation

</details>

---

## The 13 Smart Tools

Every tool is accessible via `POST /api/tools/{tool_name}` and through the web UI.

| Tool | What it does | Key options |
|------|-------------|-------------|
| **Summary** | Hierarchical document summarization | Length (brief/standard/detailed), style, audience, section filter |
| **Classify** | Auto-categorize documents | Classify by type, topics, sentiment, or urgency |
| **Extract** | Pull structured data from unstructured text | Tables, key-value pairs, entities, financial data, custom schemas |
| **Compare** | Side-by-side document comparison | Structural, semantic, or regulatory alignment |
| **Q&A** | Generate question-answer pairs | Count, difficulty level, format (quiz/FAQ/training) |
| **Split** | Break documents into chunks | By heading, by size, or by semantic boundary |
| **Redaction** | Detect and redact PII | Auto or selective mode, configurable entity types, audit trail |
| **Anonymize** | Replace entities with consistent pseudonyms | Entity types, consistency key for reproducibility |
| **Fill** | Auto-populate templates from source docs | Field mapping, confidence threshold |
| **Rewrite** | Transform tone, clarity, or reading level | Mode (improve/simplify/formalize/enrich), tone, reading level |
| **Translate** | Structure-preserving translation | Target language, glossary support, formatting preservation |
| **Merge** | Combine multiple documents | Concatenate, deduplicate, or synthesize |
| **Audit** | Check compliance and completeness | Compliance, style, or completeness checks against standards |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd IDP
pip install -e .
```

### 2. Configure

Set environment variables (or create a `.env` file):

```bash
# At least one LLM provider key is required
OPENAI_API_KEY=sk-...

# Or use other providers:
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
# OPENROUTER_API_KEY=...

# Security (required for production)
SESSION_SECRET=your-strong-random-secret-here

# Optional
# IDP_ADMIN_PASSWORD=change-me          # Default admin password (default: admin123)
# IDP_DEFAULT_MODEL=gpt-4o-2024-11-20
# ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
# DATABASE_URL=postgresql://...         # Auto-set on Replit
```

### 3. Run

```bash
python run_server.py
```

- Web UI: http://localhost:5000
- Swagger docs: http://localhost:5000/docs

### Docker

```bash
docker compose up --build
```

---

## Optional Dependencies

```bash
pip install -e ".[postgres]"   # PostgreSQL support
pip install -e ".[ocr]"       # Image/OCR processing
pip install -e ".[graph]"     # NetworkX graph analytics
pip install -e ".[all]"       # Everything above
pip install -e ".[dev]"       # pytest, ruff, coverage
```

---

## How It Works

### Document Indexing

When you upload a document, IDP Kit:

1. **Parses** it (PDF, DOCX, HTML, XLSX, PPTX, or image)
2. **Builds a tree index** — a hierarchical structure of sections, summaries, and page ranges (using LLM reasoning, not vector similarity)
3. **Extracts a knowledge graph** — entities, relationships, and cross-document links
4. **Stores everything** in the database for fast retrieval

### File Upload Flow

IDP Kit supports two upload modes:

- **Direct upload (GCS)** — Client requests a signed PUT URL via `POST /api/documents/upload-url`, uploads the file directly to cloud storage, then confirms via `POST /api/documents/{id}/confirm-upload`. The server never receives the file body, supporting files up to 50 MB.
- **Server-proxied upload (local)** — Traditional multipart upload via `POST /api/documents/` for local storage deployments.

The frontend auto-detects the available mode via `GET /api/documents/upload-mode`.

### Retrieval

Queries go through **LLM-guided tree search**: the model reasons about which sections are relevant at each level of the hierarchy, drilling deeper into promising subtrees. This is fundamentally different from vector similarity search — it uses **reasoning** instead of **similarity**.

When the knowledge graph is available, search results are **augmented** with graph-linked sections — finding related content that shares entities even if the wording is different.

### AI Agent (IDA)

The agent has access to 6 tools:

| Tool | What it does |
|------|-------------|
| `search_document` | LLM-guided tree search for relevant sections |
| `list_documents` | List available documents |
| `summarize_section` | Summarize a specific section |
| `extract_data` | Extract structured data (tables, entities, key facts, etc.) |
| `query_graph` | Query the knowledge graph (find entities, mentions, relationships, cross-doc links) |
| `find_cross_references` | Find all sections across all documents mentioning a topic |

The agent decides which tools to call, executes them, processes results, and responds in natural language. Conversations are persisted and can be continued across sessions.

### Batch Processing

The batch processor supports concurrent multi-document processing with a two-pass AI-powered flow:

1. **Pass 1 (Schema Generation)** — Analyzes your plain-text instructions (and optional reference documents) to generate a custom JSON schema tailored to your task. No hardcoded fields.
2. **Pass 2 (Structured Processing)** — Uses the generated schema as a structured output constraint when processing each target document, ensuring consistent output across all documents.

---

## Knowledge Graph

IDP Kit automatically builds a knowledge graph when documents are indexed.

### What gets extracted

- **Entities**: People, organizations, locations, concepts, terms, regulations, products, events, dates, metrics
- **Relationships**: Co-occurrence, references, defines, extends, contrasts
- **Cross-document links**: Same entities across different documents are automatically linked

### What it enables

- **"Where is X mentioned?"** — find every section across every document that mentions an entity
- **Related sections** — discover sections discussing the same entities, even with different wording
- **Document connections** — see which documents share entities and are related
- **Graph-augmented search** — tree search results expanded with graph-linked sections
- **Agent intelligence** — the AI agent answers relationship and cross-reference questions

### Graph API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graph/entities` | Search entities by name/type |
| `GET` | `/api/graph/entities/{id}` | Entity details with mentions and edges |
| `GET` | `/api/graph/entities/{id}/mentions` | All sections where entity appears |
| `GET` | `/api/graph/entities/{id}/neighbors` | Connected entities |
| `GET` | `/api/graph/documents/{doc_id}/entities` | All entities in a document |
| `GET` | `/api/graph/documents/{doc_id}/links` | Cross-document links |
| `GET` | `/api/graph/documents/{doc_id}/summary` | Graph statistics |
| `POST` | `/api/graph/documents/{doc_id}/build` | Build graph retroactively |
| `GET` | `/api/graph/documents/{doc_id}/visualization` | Nodes + edges JSON for visualization |

---

## API Overview

IDP Kit exposes 60+ REST endpoints across 13 routers:

| Router | Prefix | Endpoints |
|--------|--------|-----------|
| Auth | `/api/auth` | Register, login, token refresh, API keys |
| Documents | `/api/documents` | Upload (direct + multipart), list, get, delete, download |
| Indexing | `/api/indexing` | Trigger and monitor indexing jobs |
| Jobs | `/api/jobs` | Job status and history |
| Retrieval | `/api/retrieval` | Tree search, context building, multi-doc search |
| Agent | `/api/agent` | Chat with the AI agent, conversation management |
| Tools | `/api/tools` | Run any of the 13 smart tools |
| Generation | `/api/generation` | Generate DOCX/Markdown from templates |
| Processing | `/api/processing` | Pipelines, entity extraction, conversion |
| Batch | `/api/batch` | Batch processing jobs, templates, upload-and-process |
| Admin | `/api/admin` | User management, role promotion/demotion, rate limit config |
| Settings | `/api/settings` | LLM providers, models, prompt management |
| Graph | `/api/graph` | Knowledge graph queries and visualization |
| Tags | `/api/tags` | Document tagging and grouping |
| Plugins | `/api/plugins` | List and manage plugins |

Full interactive docs at `/docs` (Swagger) or `/redoc`.

### Example: Upload, index, and query

```bash
# 1. Upload
curl -X POST http://localhost:5000/api/documents/ \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@report.pdf"

# 2. Index (returns a job ID)
curl -X POST http://localhost:5000/api/indexing/documents/DOC_ID/index \
  -H "Authorization: Bearer TOKEN"

# 3. Search
curl -X POST http://localhost:5000/api/retrieval/search \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "query": "What are the key findings?"}'

# 4. Chat with agent
curl -X POST http://localhost:5000/api/agent/chat \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Compare the conclusions across all documents", "document_ids": ["DOC1", "DOC2"]}'
```

### Example: Direct upload (large files)

```bash
# 1. Request a signed upload URL
curl -X POST http://localhost:5000/api/documents/upload-url \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename": "large-report.pdf", "content_type": "application/pdf", "size": 45000000}'

# Response: {"upload_url": "https://storage.googleapis.com/...", "doc_id": "...", "uses_signed_url": true}

# 2. Upload directly to storage
curl -X PUT "SIGNED_URL" \
  -H "Content-Type: application/pdf" \
  --data-binary @large-report.pdf

# 3. Confirm upload
curl -X POST http://localhost:5000/api/documents/DOC_ID/confirm-upload \
  -H "Authorization: Bearer TOKEN"
```

### Example: Run a smart tool

```bash
# Summarize a document
curl -X POST http://localhost:5000/api/tools/summary \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "length": "brief", "audience": "executive"}'

# Extract structured data
curl -X POST http://localhost:5000/api/tools/extract \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "type": "key_value"}'

# Redact PII
curl -X POST http://localhost:5000/api/tools/redaction \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "mode": "auto"}'

# Translate
curl -X POST http://localhost:5000/api/tools/translate \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "target_language": "Spanish"}'
```

### Example: Admin operations

```bash
# Promote a user to admin
curl -X POST http://localhost:5000/api/admin/users/USER_ID/promote \
  -H "Authorization: Bearer TOKEN"

# Update rate limits
curl -X PUT http://localhost:5000/api/admin/rate-limits \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"agent_chat": "50/minute", "batch_create": "20/minute"}'

# Get current rate limits
curl http://localhost:5000/api/admin/rate-limits \
  -H "Authorization: Bearer TOKEN"
```

### Example: Knowledge graph queries

```bash
# Search entities
curl http://localhost:5000/api/graph/entities?name=algorithm&type=CONCEPT \
  -H "Authorization: Bearer TOKEN"

# Find cross-document links
curl http://localhost:5000/api/graph/documents/DOC_ID/links \
  -H "Authorization: Bearer TOKEN"

# Build graph for an existing document
curl -X POST http://localhost:5000/api/graph/documents/DOC_ID/build \
  -H "Authorization: Bearer TOKEN"
```

---

## Supported Formats

| Format | Extensions | Parser | Indexer |
|--------|-----------|--------|--------|
| PDF | `.pdf` | PDF Parser | PDF Indexer (full tree with LLM reasoning) |
| Word | `.docx`, `.doc` | DOCX Parser | DOCX Indexer (heading-based hierarchy) |
| HTML | `.html`, `.htm` | HTML Parser | HTML Indexer (semantic heading structure) |
| Markdown | `.md`, `.markdown` | — | Markdown Indexer (full tree with LLM reasoning) |
| Spreadsheet | `.xlsx`, `.xls`, `.csv` | Spreadsheet Parser | Generic Indexer (sheet-per-node) |
| Presentation | `.pptx`, `.ppt` | PPTX Parser | Generic Indexer (slide-per-node) |
| Image | `.png`, `.jpg`, `.tiff`, `.bmp`, `.webp`, `.gif` | Image/OCR Parser | Generic Indexer |

**Note:** SVG uploads are blocked for security (potential XSS via embedded JavaScript).

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SESSION_SECRET` | Yes (prod) | JWT signing key. If not set, an ephemeral random key is used (sessions won't survive restarts). |
| `DATABASE_URL` | No | PostgreSQL connection string. Falls back to SQLite if not set. Auto-set on Replit. |
| `DEFAULT_OBJECT_STORAGE_BUCKET_ID` | No | GCS bucket ID for file storage. Falls back to local `./storage`. |
| `PRIVATE_OBJECT_DIR` | No | Private directory prefix in GCS bucket. |
| `OPENAI_API_KEY` | No* | OpenAI API key. *At least one LLM provider key is required. |
| `ANTHROPIC_API_KEY` | No | Anthropic API key. |
| `GOOGLE_API_KEY` | No | Google/Gemini API key. |
| `OPENROUTER_API_KEY` | No | OpenRouter API key. |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`). |
| `IDP_ADMIN_PASSWORD` | No | Default admin password (default: `admin123`). |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS allowed origins. |
| `IDP_DEFAULT_MODEL` | No | Default LLM model (default: `gpt-4o`). |

---

## CLI: Tree Index Generation

Generate tree indices directly from the command line:

```bash
# PDF
python run_pageindex.py --pdf_path /path/to/document.pdf

# Markdown
python run_pageindex.py --md_path /path/to/document.md
```

<details>
<summary>Optional parameters</summary>

```
--model                 LLM model (default: gpt-4o-2024-11-20)
--toc-check-pages       Pages to check for TOC (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node IDs (yes/no, default: yes)
--if-add-node-summary   Add summaries (yes/no, default: yes)
--if-add-doc-description Add description (yes/no, default: yes)
```
</details>

---

## Project Structure

```
idpkit/
  api/             FastAPI app factory and routes (15 routers, 60+ endpoints)
    deps.py        Shared dependencies: auth, storage, rate limiter
    routes/        API route modules (auth, documents, indexing, agent, batch, admin, etc.)
  web/             Jinja2 + Alpine.js frontend templates
  db/              SQLAlchemy 2.0 async models, session, seeding, migrations
  engine/          Tree indexing core — PDF/Markdown tree index generation
  core/            LLMClient (LiteLLM), schemas, storage (GCS + local), exceptions
  graph/           Knowledge Graph — entities, edges, builder, linker, queries
  indexing/        Multi-format indexers (PDF, DOCX, HTML, XLSX, PPTX, Image)
  parsing/         Document parsers (6 formats)
  retrieval/       Tree search, context builder, multi-doc search
  agent/           AI agent with tool-calling loop
  tools/           13 Smart Tools
  generation/      DOCX/Markdown generators, templates
  processing/      Pipelines, entity extraction, summarization, comparison
  batch/           Batch processing runner with concurrent execution
  plugins/         Plugin system (entry-point + directory discovery)
pageindex/         Standalone PageIndex library
```

### Database schema

| Table | Purpose |
|-------|---------|
| `users` | Authentication, roles (superadmin/admin/user), API keys |
| `documents` | Uploaded documents with tree index, metadata, and status (indexed) |
| `jobs` | Background job tracking (indexing, processing) |
| `prompts` | User-saved prompt templates |
| `templates` | Document generation templates |
| `processing_templates` | Batch processing templates |
| `batch_jobs` | Batch processing job tracking |
| `batch_items` | Individual items within a batch job |
| `conversations` | Agent chat conversation metadata |
| `conversation_messages` | Agent conversation message history |
| `entities` | Knowledge graph entity registry |
| `entity_mentions` | Entity-to-section links |
| `graph_edges` | Entity relationships and cross-document links |
| `tags` | Document tags |
| `document_tags` | Many-to-many document-tag junction |
| `system_settings` | Admin-configurable settings (rate limits, etc.) |

---

## Key Dependencies

- FastAPI, Uvicorn, SQLAlchemy (async), asyncpg
- OpenAI, LiteLLM, tiktoken
- PyMuPDF, PyPDF2, python-docx, beautifulsoup4
- Jinja2, Pydantic, httpx
- passlib + bcrypt (auth), python-jose (JWT)
- slowapi (rate limiting)
- NetworkX (optional, for graph analytics)

---

## Acknowledgements

IDP Kit's document indexing and retrieval engine is built on [PageIndex](https://github.com/VectifyAI/PageIndex) by [Vectify AI](https://vectify.ai) — a vectorless, reasoning-based RAG system that uses hierarchical tree indexing and LLM-guided tree search instead of vector similarity.

---

## License

MIT
