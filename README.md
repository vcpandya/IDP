# IDP Kit — Intelligent Document Processing Toolkit & AI Agent

**IDP Kit** is a full-stack Python toolkit for intelligent document processing. It combines tree-based document indexing, a knowledge graph, an AI agent with tool-calling, 13 smart document tools, and a web UI — all in one package.

Upload PDFs, DOCX, HTML, spreadsheets, presentations, or images. IDP Kit indexes them into a hierarchical tree structure, extracts a knowledge graph of entities and relationships, and lets you query everything through an AI agent or REST API.

## Features

| Category | What you get |
|----------|-------------|
| **Document Indexing** | Hierarchical tree index (powered by [PageIndex](https://github.com/VectifyAI/PageIndex)) — no vector DB, no chunking |
| **Knowledge Graph** | Automatic entity extraction, cross-document linking, relationship discovery |
| **AI Agent** | Tool-calling agent (6 tools) — search, summarize, extract, graph queries, cross-references |
| **Smart Tools** | 13 tools: Summary, Classify, Extract, Compare, Q&A, Split, Redaction, Anonymize, Fill, Rewrite, Translate, Merge, Audit |
| **Multi-Format** | PDF, DOCX, HTML, XLSX/CSV, PPTX, Images (OCR) |
| **Multi-Provider LLM** | OpenAI, Anthropic, Gemini, Ollama, OpenRouter — all via [LiteLLM](https://github.com/BerriAI/litellm) |
| **Web UI** | FastAPI + Jinja2 + HTMX + Alpine.js |
| **REST API** | 54 endpoints across 11 routers |
| **Plugin System** | Entry-point + directory-based plugin discovery |
| **Database** | SQLAlchemy 2.0 async — SQLite (dev) / PostgreSQL (prod) |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd IDP
pip install -e .
```

### 2. Configure

Create a `.env` file:

```bash
# At least one LLM provider key is required
OPENAI_API_KEY=sk-...

# Or use other providers:
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
# OPENROUTER_API_KEY=...

# Optional
# IDP_DEFAULT_MODEL=gpt-4o-2024-11-20
# IDP_DATABASE_URL=sqlite+aiosqlite:///./idpkit.db
# IDP_SECRET_KEY=change-me-in-production
# IDP_STORAGE_PATH=./storage
```

### 3. Run

```bash
python run_server.py
```

- Web UI: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

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

### Retrieval

Queries go through **LLM-guided tree search**: the model reasons about which sections are relevant at each level of the hierarchy, drilling deeper into promising subtrees. This is fundamentally different from vector similarity search — it uses **reasoning** instead of **similarity**.

When the knowledge graph is available, search results are **augmented** with graph-linked sections — finding related content that shares entities even if the wording is different.

### AI Agent

The agent has access to 6 tools:

| Tool | What it does |
|------|-------------|
| `search_document` | LLM-guided tree search for relevant sections |
| `list_documents` | List available documents |
| `summarize_section` | Summarize a specific section |
| `extract_data` | Extract structured data (tables, entities, key facts, etc.) |
| `query_graph` | Query the knowledge graph (find entities, mentions, relationships, cross-doc links) |
| `find_cross_references` | Find all sections across all documents mentioning a topic |

The agent decides which tools to call, executes them, processes results, and responds in natural language.

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

### Retroactive building

For documents indexed before the graph feature existed:

```bash
curl -X POST http://localhost:8000/api/graph/documents/{doc_id}/build \
  -H "Authorization: Bearer TOKEN"
```

---

## API Overview

IDP Kit exposes 54 REST endpoints across 11 routers:

| Router | Prefix | Endpoints |
|--------|--------|-----------|
| Auth | `/api/auth` | Register, login, token refresh, API keys |
| Documents | `/api/documents` | Upload, list, get, delete documents |
| Indexing | `/api/indexing` | Trigger and monitor indexing jobs |
| Jobs | `/api/jobs` | Job status and history |
| Retrieval | `/api/retrieval` | Tree search, context building, multi-doc search |
| Agent | `/api/agent` | Chat with the AI agent |
| Tools | `/api/tools` | Run any of the 13 smart tools |
| Generation | `/api/generation` | Generate DOCX/Markdown from templates |
| Processing | `/api/processing` | Pipelines, entity extraction, conversion |
| Plugins | `/api/plugins` | List and manage plugins |
| Graph | `/api/graph` | Knowledge graph queries and visualization |

Full interactive docs at `/docs` (Swagger) or `/redoc`.

### Example: Upload, index, and query

```bash
# 1. Upload
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@report.pdf"

# 2. Index (returns a job ID)
curl -X POST http://localhost:8000/api/indexing/index \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID"}'

# 3. Search
curl -X POST http://localhost:8000/api/retrieval/search \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "DOC_ID", "query": "What are the key findings?"}'

# 4. Chat with agent
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Compare the conclusions across all documents", "document_ids": ["DOC1", "DOC2"]}'
```

---

## CLI: PageIndex Tree Generation

You can also use the PageIndex engine directly from the command line to generate tree indices:

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
  engine/          PageIndex core — PDF/Markdown tree indexing
  core/            LLMClient (LiteLLM), schemas, storage, exceptions
  db/              SQLAlchemy 2.0 async models + session management
  graph/           Knowledge Graph — entities, edges, builder, linker, queries
  indexing/        Multi-format indexers (PDF, DOCX, HTML, XLSX, PPTX, Image)
  parsing/         Document parsers (6 formats)
  retrieval/       Tree search, context builder, multi-doc search
  agent/           AI agent with tool-calling loop
  tools/           13 Smart Tools
  generation/      DOCX/Markdown generators, templates
  processing/      Pipelines, entity extraction, summarization, comparison
  plugins/         Plugin system (entry-point + directory discovery)
  api/             FastAPI routes (11 routers, 54 endpoints)
  web/             Jinja2 + HTMX + Alpine.js frontend
pageindex/         Backward-compatible shim for PageIndex imports
```

### Database schema

| Table | Purpose |
|-------|---------|
| `users` | Authentication and authorization |
| `documents` | Uploaded documents with tree index and metadata |
| `jobs` | Background job tracking |
| `prompts` | User-saved prompt templates |
| `templates` | Document generation templates |
| `conversation_messages` | Agent conversation history |
| `entities` | Knowledge graph entity registry |
| `entity_mentions` | Entity-to-section links |
| `graph_edges` | Entity relationships and cross-document links |

---

## Acknowledgements

IDP Kit's document indexing and retrieval engine is built on [PageIndex](https://github.com/VectifyAI/PageIndex) by [Vectify AI](https://vectify.ai) — a vectorless, reasoning-based RAG system that uses hierarchical tree indexing and LLM-guided tree search instead of vector similarity. PageIndex achieved 98.7% accuracy on [FinanceBench](https://arxiv.org/abs/2311.11944), demonstrating the power of reasoning-based retrieval over traditional vector approaches.

---

## License

MIT
