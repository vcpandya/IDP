# IDP Kit — API Integration Guide

This document describes how an external system can integrate with the IDP Kit REST API. Every endpoint listed below is served over HTTPS and follows standard REST conventions. All request and response bodies use JSON unless otherwise noted.

---

## Table of Contents

1. [Base URL](#1-base-url)
2. [Authentication](#2-authentication)
3. [Common Conventions](#3-common-conventions)
4. [Documents](#4-documents)
5. [Tags](#5-tags)
6. [Indexing](#6-indexing)
7. [Retrieval (RAG)](#7-retrieval-rag)
8. [AI Agent (Chat)](#8-ai-agent-chat)
9. [Smart Tools](#9-smart-tools)
10. [Knowledge Graph](#10-knowledge-graph)
11. [Batch Processing](#11-batch-processing)
12. [Jobs](#12-jobs)
13. [Error Reference](#13-error-reference)
14. [Typical Integration Workflow](#14-typical-integration-workflow)

---

## 1. Base URL

All endpoints are prefixed with the deployment host, for example:

```
https://your-idpkit-instance.example.com
```

Replace this with the actual URL of the IDP Kit deployment you are integrating with.

---

## 2. Authentication

IDP Kit supports two authentication methods. Every endpoint (except registration and login) requires one of them.

### 2.1 JWT Bearer Token

Obtain a token by calling the login endpoint. Include it in subsequent requests via the `Authorization` header.

**Login**

```
POST /api/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response (200)**

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

Use the token in all further requests:

```
Authorization: Bearer eyJhbGciOi...
```

Tokens expire after the `expires_in` period (in seconds). Re-authenticate when a `401` response is received.

### 2.2 API Key

API keys are long-lived credentials ideal for server-to-server integrations. Generate one after logging in:

```
POST /api/auth/apikey
Authorization: Bearer <jwt_token>
```

**Response (200)**

```json
{
  "api_key": "idpk_abc123..."
}
```

> **Important:** Store the API key securely when it is generated. Calling this endpoint again will replace the previous key.

Include the key in requests via the `X-API-Key` header:

```
X-API-Key: idpk_abc123...
```

### 2.3 Register a New Account

```
POST /api/auth/register
Content-Type: application/json

{
  "username": "integrator",
  "password": "secureP@ss1",
  "email": "integrator@example.com"
}
```

**Response (201)**

```json
{
  "id": "uuid",
  "username": "integrator",
  "email": "integrator@example.com",
  "role": "user",
  "is_active": 0,
  "api_key": null,
  "created_at": "2025-01-15T10:00:00Z"
}
```

> **Note:** New accounts are created with `is_active: 0` and require admin approval before they can log in.

### 2.4 Get Current User

```
GET /api/auth/me
Authorization: Bearer <token>
```

Returns the authenticated user's profile.

---

## 3. Common Conventions

| Convention | Detail |
|---|---|
| **Content-Type** | `application/json` for all JSON endpoints; `multipart/form-data` for file uploads. |
| **IDs** | All resource IDs are UUIDs (strings). |
| **Pagination** | List endpoints accept `skip` (offset, default 0) and `limit` (max rows, default 20). |
| **Timestamps** | ISO 8601 format, UTC. |
| **Errors** | JSON body with a `detail` field describing the error (see [Error Reference](#12-error-reference)). |
| **File size limit** | 50 MB per upload. |

### Rate Limiting

Certain endpoints enforce per-user rate limits to protect system resources. When a limit is exceeded, the server responds with HTTP `429 Too Many Requests`.

| Endpoint | Default Limit |
|---|---|
| `POST /api/agent/chat` | 30 requests per minute |
| `POST /api/batch/` | 10 requests per minute |

Other endpoints do not currently enforce explicit rate limits, but integrators should still implement reasonable request pacing to avoid overloading the server.

**Handling 429 responses:**

- Inspect the `Retry-After` header (if present) for the number of seconds to wait before retrying.
- If no `Retry-After` header is returned, implement exponential backoff starting at 1 second (1s → 2s → 4s → ...).
- Rate limits are applied per authenticated user. Different users have independent quotas.
- Administrators can adjust these limits via the admin settings API.

---

## 4. Documents

### 4.1 Upload a Document

Upload a file directly via multipart form data.

```bash
curl -X POST https://host/api/documents/ \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@/path/to/report.pdf"
```

**Supported formats:** PDF, DOCX, DOC, MD, HTML, XLSX, XLS, CSV, PPTX, PPT, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP.

**Response (201)**

```json
{
  "id": "doc-uuid",
  "filename": "report.pdf",
  "format": "pdf",
  "file_size": 2048576,
  "page_count": 24,
  "status": "uploaded",
  "owner_id": "user-uuid",
  "created_at": "2025-01-15T10:05:00Z",
  "tags": []
}
```

### 4.2 Upload via Signed URL (Large Files)

For large files, request a signed upload URL and upload directly to cloud storage.

**Step 1 — Request upload URL**

```
POST /api/documents/upload-url
Authorization: Bearer <token>
Content-Type: application/json

{
  "filename": "large-report.pdf",
  "content_type": "application/pdf",
  "size": 40000000
}
```

**Response (200)**

```json
{
  "upload_url": "https://storage.googleapis.com/...",
  "doc_id": "doc-uuid",
  "storage_key": "user-uuid/doc-uuid/original.pdf",
  "uses_signed_url": true
}
```

**Step 2 — Upload the file** to the returned `upload_url` using an HTTP `PUT`.

**Step 3 — Confirm the upload**

```
POST /api/documents/{doc_id}/confirm-upload
Authorization: Bearer <token>
```

If `uses_signed_url` is `false`, upload the file content to the fallback endpoint instead:

```
POST /api/documents/{doc_id}/upload-content
Authorization: Bearer <token>
```

### 4.3 List Documents

```
GET /api/documents/?skip=0&limit=20
```

**Response (200)**

```json
{
  "items": [ /* array of document objects */ ],
  "total": 42,
  "skip": 0,
  "limit": 20
}
```

### 4.4 Get Document Details

```
GET /api/documents/{doc_id}
```

Returns the full document object including `tree_index` (if indexed).

### 4.5 Download Document

```
GET /api/documents/{doc_id}/download
```

Returns the raw file bytes with the appropriate `Content-Type` header.

### 4.6 Delete Document

```
DELETE /api/documents/{doc_id}
```

Deletes the document record and its stored files. Returns `{"detail": "Document deleted"}`.

### 4.7 Auto-Tag a Document

Use AI to suggest and optionally apply tags:

```
POST /api/documents/{doc_id}/auto-tag
Content-Type: application/json

{
  "apply": true
}
```

**Response (200)**

```json
{
  "document_id": "doc-uuid",
  "suggestions": [
    { "name": "Financial Report", "existing_id": null, "confidence": 0.92 }
  ],
  "applied": [
    { "tag_id": "tag-uuid", "name": "Financial Report" }
  ]
}
```

---

## 5. Tags

Tags let you organise documents into groups (e.g., by project, department, or document type). Tags are referenced in agent chat (`tag_ids`), graph visualisation, and auto-tagging.

### 5.1 Create a Tag

```
POST /api/tags/
Content-Type: application/json

{
  "name": "Q3 Reports",
  "color": "#4f46e5",
  "description": "All Q3 2024 financial reports"
}
```

**Response (201)**

```json
{
  "id": "tag-uuid",
  "name": "Q3 Reports",
  "color": "#4f46e5",
  "description": "All Q3 2024 financial reports",
  "document_count": 0,
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:00:00Z"
}
```

### 5.2 List Tags

```
GET /api/tags/
```

Returns an array of tag objects with document counts.

### 5.3 Get Tag Details

```
GET /api/tags/{tag_id}
```

Returns the tag with its list of associated documents.

### 5.4 Update a Tag

```
PATCH /api/tags/{tag_id}
Content-Type: application/json

{
  "name": "Q3 Reports 2024",
  "color": "#2563eb"
}
```

All fields are optional; only provided fields are updated.

### 5.5 Delete a Tag

```
DELETE /api/tags/{tag_id}
```

### 5.6 Add Documents to a Tag

```
POST /api/tags/{tag_id}/documents
Content-Type: application/json

{
  "document_ids": ["doc-uuid-1", "doc-uuid-2"]
}
```

### 5.7 Remove a Document from a Tag

```
DELETE /api/tags/{tag_id}/documents/{doc_id}
```

---

## 6. Indexing

Indexing builds a hierarchical tree structure from a document's content. This is a prerequisite for retrieval, agent chat with document context, and knowledge graph features.

### 6.1 Trigger Indexing

```
POST /api/indexing/documents/{doc_id}/index
Content-Type: application/json

{
  "model": "gpt-4o",
  "max_pages_per_node": 50
}
```

All fields are optional. The endpoint returns immediately with a job reference (HTTP 202).

**Response (202)**

```json
{
  "job_id": "job-uuid",
  "document_id": "doc-uuid",
  "status": "pending",
  "detail": "Indexing job queued"
}
```

### 6.2 Poll Indexing Status

```
GET /api/indexing/documents/{doc_id}/index/status?last_log_index=0
```

**Response (200)**

```json
{
  "id": "job-uuid",
  "job_type": "index",
  "status": "running",
  "progress": 45,
  "stage": "Analyzing document structure",
  "logs": [
    { "ts": "10:05:01", "level": "INFO", "msg": "Using model: gpt-4o" }
  ],
  "log_offset": 0
}
```

The `status` field progresses through: `pending` → `running` → `completed` or `failed`.

Use `last_log_index` to fetch only new log entries (pass the length of previously received logs).

### 6.3 Get the Tree Index

Once indexing is complete:

```
GET /api/indexing/documents/{doc_id}/tree
```

**Response (200)**

```json
{
  "document_id": "doc-uuid",
  "filename": "report.pdf",
  "status": "indexed",
  "tree_index": {
    "doc_name": "Annual Report 2024",
    "doc_description": "...",
    "structure": [ /* hierarchical nodes */ ]
  }
}
```

### 6.4 List Indexing Jobs for a Document

```
GET /api/indexing/documents/{doc_id}/jobs
```

Returns an array of all indexing job records for the document.

---

## 7. Retrieval (RAG)

Query an indexed document using tree-based Retrieval Augmented Generation.

```
POST /api/retrieval/documents/{doc_id}/query
Content-Type: application/json

{
  "query": "What were the total revenues in Q3?",
  "max_context_tokens": 4000
}
```

**Response (200)**

```json
{
  "answer": "According to the Q3 Financial Summary (pages 12-14), total revenues were $4.2M...",
  "sources": [
    {
      "node_id": "node-uuid",
      "title": "Q3 Financial Summary",
      "start_page": 12,
      "end_page": 14,
      "summary": "This section covers quarterly revenue..."
    }
  ],
  "query": "What were the total revenues in Q3?",
  "document_id": "doc-uuid"
}
```

The document must have status `indexed` (i.e., indexing must be completed first).

---

## 8. AI Agent (Chat)

The agent is a conversational AI that can search documents, query the knowledge graph, perform web searches, and use smart tools.

### 8.1 Send a Message (Stateless)

Send a one-off message without conversation history:

```
POST /api/agent/chat
Content-Type: application/json

{
  "message": "Summarise the key findings from my uploaded reports",
  "document_ids": ["doc-uuid-1", "doc-uuid-2"]
}
```

You may also pass `tag_ids` to include all documents with specific tags.

**Response (200)**

```json
{
  "response": "Based on your reports, the key findings are...",
  "conversation_id": null,
  "tool_calls": [
    {
      "name": "search_document",
      "args": { "document_id": "doc-uuid-1", "query": "key findings" },
      "result": { "results": [ /* matching nodes */ ] }
    }
  ],
  "sources": [
    {
      "document_id": "doc-uuid-1",
      "filename": "report-2024.pdf",
      "title": "Executive Summary",
      "start_page": 1,
      "end_page": 3
    }
  ],
  "source_type": "documents",
  "search_attempts": [
    {
      "document_id": "doc-uuid-1",
      "filename": "report-2024.pdf",
      "query": "key findings",
      "results_found": 3,
      "status": "found"
    }
  ]
}
```

`source_type` values: `documents`, `general_knowledge`, `mixed`, `web`.

### 8.2 Continue a Conversation

To persist messages across turns, first create a conversation, then pass its ID:

```
POST /api/agent/conversations
Content-Type: application/json

{ "title": "Revenue Analysis" }
```

**Response (201)** returns the new conversation object including its `id`. Use that ID in subsequent chat requests:

```json
{
  "message": "Can you go into more detail on revenue trends?",
  "conversation_id": "conv-uuid",
  "document_ids": ["doc-uuid-1"]
}
```

The agent will load all prior messages from the conversation for context. If `conversation_id` is omitted or `null`, the chat is stateless and no messages are persisted.

### 8.3 Conversation Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agent/conversations` | List conversations (most recent first, max 50) |
| `POST` | `/api/agent/conversations` | Create a new conversation (`{"title": "My conversation"}`) |
| `GET` | `/api/agent/conversations/{id}` | Get conversation with full message history |
| `PATCH` | `/api/agent/conversations/{id}` | Rename a conversation (`{"title": "New name"}`) |
| `DELETE` | `/api/agent/conversations/{id}` | Delete a conversation (HTTP 204) |

---

## 9. Smart Tools

Smart Tools perform specific AI-powered operations on individual documents (summarisation, extraction, classification, etc.).

### 9.1 List Available Tools

```
GET /api/tools/
```

**Response (200)**

```json
{
  "tools": [
    {
      "name": "smart_summary",
      "display_name": "Smart Summary",
      "description": "Generates hierarchical summaries using the document tree structure",
      "options_schema": { /* JSON Schema for tool-specific options */ }
    }
  ],
  "total": 13
}
```

### 9.2 Execute a Tool

```
POST /api/tools/smart_summary
Content-Type: application/json

{
  "document_id": "doc-uuid",
  "options": {
    "length": "detailed"
  },
  "model": "gpt-4o"
}
```

The `options` object varies by tool. Refer to the `options_schema` returned by the list endpoint.

**Response (200)**

```json
{
  "tool_name": "smart_summary",
  "status": "success",
  "data": {
    "summary": "This document covers..."
  },
  "output_file": null,
  "error": null
}
```

---

## 10. Knowledge Graph

After indexing, IDP Kit can build a knowledge graph of entities (people, organisations, concepts) and their relationships across documents.

### 10.1 Build Graph for a Document

```
POST /api/graph/documents/{doc_id}/build
```

Triggers entity extraction and cross-document linking. The document must already be indexed.

### 10.2 Search Entities

```
GET /api/graph/entities?name=Acme&entity_type=Organization&limit=50
```

**Response (200)**

```json
[
  {
    "id": "entity-uuid",
    "canonical_name": "Acme Corporation",
    "entity_type": "Organization",
    "description": "A multinational technology company",
    "aliases": ["Acme Corp", "ACME"]
  }
]
```

### 10.3 Entity Details

```
GET /api/graph/entities/{entity_id}
```

Returns the entity with all its mentions (linked to specific document sections) and edges (relationships to other entities).

### 10.4 Entity Neighbours

```
GET /api/graph/entities/{entity_id}/neighbors?relation_type=works_for&limit=50
```

Returns entities connected to the given entity, with their relationship edges.

### 10.5 Document-Level Graph Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/graph/documents/{doc_id}/entities` | List entities found in a document |
| `GET` | `/api/graph/documents/{doc_id}/links` | Find cross-document connections via shared entities |
| `GET` | `/api/graph/documents/{doc_id}/summary` | Get graph statistics (entity count, edge count, types) |
| `GET` | `/api/graph/documents/{doc_id}/visualization` | Get nodes + edges JSON for graph visualisation |

### 10.6 Multi-Document Visualisation

```
GET /api/graph/visualization?doc_ids=uuid1,uuid2&limit=1000
```

Or filter by tag:

```
GET /api/graph/visualization?tag_id=tag-uuid&limit=1000
```

### 10.7 Export Full Graph

```
GET /api/graph/export?doc_ids=uuid1,uuid2&format=json
```

Supported `format` values: `json`, `csv_entities`, `csv_relationships`. Returns a downloadable file.

### 10.8 List Entity Types

```
GET /api/graph/entity-types
```

Returns an array of all distinct entity type strings (e.g., `["Concept", "Organization", "Person"]`).

---

## 11. Batch Processing

Process multiple documents in parallel using templates.

### 11.1 Templates

Templates define reusable processing configurations.

**Create a template**

```
POST /api/batch/templates
Content-Type: application/json

{
  "name": "Invoice Extraction",
  "description": "Extract line items from invoices",
  "tool_name": "smart_extract",
  "tool_options": { "fields": ["vendor", "total", "date"] },
  "output_format": "json",
  "model": "gpt-4o"
}
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/batch/templates` | Create a template |
| `GET` | `/api/batch/templates` | List your templates (and public ones) |
| `GET` | `/api/batch/templates/{id}` | Get template details |
| `PUT` | `/api/batch/templates/{id}` | Update a template |
| `DELETE` | `/api/batch/templates/{id}` | Delete a template |

**AI-generated template from a sample document**

```bash
curl -X POST https://host/api/batch/templates/analyze \
  -H "X-API-Key: YOUR_KEY" \
  -F "sample_file=@/path/to/sample-invoice.pdf" \
  -F "template_name=Invoice Template"
```

### 11.2 Start a Batch Job

```
POST /api/batch/
Content-Type: application/json

{
  "template_id": "template-uuid",
  "document_ids": ["doc-uuid-1", "doc-uuid-2", "doc-uuid-3"],
  "concurrency": 3,
  "model": "gpt-4o"
}
```

Alternatively, specify `tool_name`, `prompt`, and `options` directly instead of `template_id`.

**Response (202)**

```json
{
  "id": "batch-uuid",
  "status": "pending",
  "total_items": 3,
  "completed_items": 0,
  "failed_items": 0,
  "concurrency": 3
}
```

### 11.3 Monitor a Batch Job

```
GET /api/batch/{batch_id}
```

Returns full batch status including per-item results.

### 11.4 List Batch Jobs

```
GET /api/batch/?skip=0&limit=20
```

### 11.5 Cancel a Batch Job

```
POST /api/batch/{batch_id}/cancel
```

### 11.6 Delete a Batch Job

```
DELETE /api/batch/{batch_id}
```

### 11.7 Convert Plain-Text Options to JSON

Convert free-text processing instructions into structured options using AI:

```
POST /api/batch/convert-options
Content-Type: application/json

{
  "text": "Extract vendor name, invoice date, line items with quantities and prices, and total amount",
  "tool_name": "custom"
}
```

**Response (200)**

```json
{
  "options": {
    "vendor_name": "string",
    "invoice_date": "string",
    "line_items": [{ "description": "string", "quantity": "number", "price": "number" }],
    "total_amount": "number"
  },
  "schema_used": { /* generated JSON Schema */ },
  "raw_text": "Extract vendor name, invoice date..."
}
```

---

## 12. Jobs

All long-running operations (indexing, batch processing) create job records that can be tracked.

### 12.1 List Jobs

```
GET /api/jobs/
```

**Response (200)**

```json
{
  "items": [ /* array of job objects */ ]
}
```

### 12.2 Get Job Details

```
GET /api/jobs/{job_id}
```

**Response (200)**

```json
{
  "id": "job-uuid",
  "job_type": "index",
  "status": "completed",
  "progress": 100,
  "document_id": "doc-uuid",
  "result": { /* output data */ },
  "created_at": "2025-01-15T10:05:00Z",
  "completed_at": "2025-01-15T10:06:30Z"
}
```

### 12.3 Stream Job Progress (SSE)

For real-time progress updates, connect to the Server-Sent Events stream:

```
GET /api/jobs/{job_id}/stream
```

The stream emits `progress` events:

```
event: progress
data: {"job_id": "job-uuid", "status": "running", "progress": 45}

event: progress
data: {"job_id": "job-uuid", "status": "completed", "progress": 100, "result": {...}}
```

The stream closes automatically when the job reaches `completed` or `failed` status.

**cURL example:**

```bash
curl -N -H "X-API-Key: YOUR_KEY" https://host/api/jobs/JOB_ID/stream
```

> **Note:** The native browser `EventSource` API does not support custom headers. For browser-based SSE, authenticate via the session cookie (set automatically on login). For server-to-server integrations, use an HTTP client that supports streaming with custom headers (e.g., `fetch`, `axios`, or `curl`).

---

## 13. Error Reference

All error responses follow the same shape:

```json
{
  "detail": "Human-readable error message"
}
```

| HTTP Status | Meaning |
|---|---|
| `400` | Bad request — invalid input, unsupported file format, or missing prerequisites (e.g., document not indexed). |
| `401` | Unauthorized — missing or invalid authentication credentials. |
| `403` | Forbidden — account pending admin approval. |
| `404` | Not found — resource does not exist or does not belong to the authenticated user. |
| `409` | Conflict — duplicate resource (e.g., username taken) or operation already in progress (e.g., indexing already running). |
| `413` | Payload too large — file exceeds 50 MB limit. |
| `429` | Too many requests — rate limit exceeded. Back off and retry (see [Rate Limiting](#rate-limiting)). |
| `500` | Internal server error — unexpected failure; contact the system administrator. |

---

## 14. Typical Integration Workflow

Below is the recommended sequence for a system integrating with IDP Kit end to end:

```
1.  Register & authenticate
      POST /api/auth/register
      POST /api/auth/login      → save the JWT token
      POST /api/auth/apikey     → save the API key for ongoing use

2.  Upload a document
      POST /api/documents/      → returns doc_id

3.  Index the document
      POST /api/indexing/documents/{doc_id}/index   → returns job_id
      GET  /api/indexing/documents/{doc_id}/index/status  (poll until status = "completed")

4.  Query the document (RAG)
      POST /api/retrieval/documents/{doc_id}/query

5.  Or chat with the AI agent
      POST /api/agent/chat   (pass document_ids for document-aware responses)

6.  Run a smart tool
      GET  /api/tools/                      → list available tools
      POST /api/tools/smart_summary         → execute on a document (example)

7.  Build & explore the knowledge graph
      POST /api/graph/documents/{doc_id}/build
      GET  /api/graph/entities?name=...
      GET  /api/graph/documents/{doc_id}/links

8.  Batch-process multiple documents
      POST /api/batch/templates             → create a template
      POST /api/batch/                      → start a batch job
      GET  /api/batch/{batch_id}            → monitor progress
```

Each step builds on the previous one. Documents must be uploaded before indexing, and indexed before querying, graph building, or tool execution.

> **Note:** The examples in this guide are illustrative. For the authoritative response schemas and field types, consult the auto-generated OpenAPI documentation at `/docs` (Swagger UI) or `/redoc` on your IDP Kit instance.
