# IDP Kit — Document Verification API

This document provides a complete reference for integrating with the IDP Kit **Document Verification** API. The verifier uses multimodal LLM analysis to check whether uploaded documents satisfy a set of natural-language expectations (e.g., "this is a signed NDA", "the invoice total exceeds $5,000").

For general authentication, conventions, and other API endpoints, see [integration_instructions.md](integration_instructions.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [Verify a Single Document](#3-verify-a-single-document)
4. [Verify Multiple Documents](#4-verify-multiple-documents)
5. [Response Field Reference](#5-response-field-reference)
6. [File Constraints](#6-file-constraints)
7. [Error Handling](#7-error-handling)
8. [Integration Tips](#8-integration-tips)

---

## 1. Overview

The verification API accepts one or more documents along with a JSON array of expectation strings. The LLM inspects each document (using vision for images/PDFs, text extraction for other formats) and returns a per-expectation verdict.

| Endpoint | Description |
|---|---|
| `POST /api/verify/single` | Verify a single file — returns a JSON result directly |
| `POST /api/verify` | Verify one or more files — single file returns JSON; multiple files stream results via SSE |

Both endpoints use `multipart/form-data` and require authentication.

---

## 2. Authentication

All verification endpoints require a valid credential. Use either method described in the main integration guide:

```
Authorization: Bearer <jwt_token>
```

or

```
X-API-Key: idpk_abc123...
```

---

## 3. Verify a Single Document

Upload one file and a JSON array of expectations.

### Request

```bash
curl -X POST https://idpai.replit.app/api/verify/single \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@/path/to/invoice.pdf" \
  -F 'expectations=["invoice total exceeds $1000", "vendor is Acme Corp", "dated within the last 90 days"]'
```

| Form field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | The document to verify (see [File Constraints](#6-file-constraints)) |
| `expectations` | string (JSON) | Yes | A JSON array of expectation strings (e.g., `'["has a signature", "dated 2025"]'`) or a single JSON string (e.g., `'"has a signature"'`) |
| `model` | string | No | LLM model override (defaults to the user's default model, or `gpt-4o`) |

### Response (200)

```json
{
  "filename": "invoice.pdf",
  "status": "not_ok",
  "matches": [
    {
      "expected": "invoice total exceeds $1000",
      "result": "ok",
      "confidence": "high",
      "details": "The invoice total is $4,250.00, which exceeds $1,000."
    },
    {
      "expected": "vendor is Acme Corp",
      "result": "not_ok",
      "confidence": "high",
      "details": "The vendor listed on the invoice is 'Widget Industries', not 'Acme Corp'."
    },
    {
      "expected": "dated within the last 90 days",
      "result": "ok",
      "confidence": "medium",
      "details": "The invoice date is 2025-11-02, which is within the last 90 days."
    }
  ],
  "summary": "2 of 3 expectations met. The vendor does not match the expected value."
}
```

The top-level `status` is intended to be `"ok"` only when **all** expectations pass; otherwise `"not_ok"`. This logic is enforced by the LLM prompt, so edge-case inconsistencies are possible — always check the individual `matches[].result` values for reliable per-expectation verdicts.

---

## 4. Verify Multiple Documents

Submit multiple files in a single request. All files are checked against the same set of expectations.

### Request

```bash
curl -X POST https://idpai.replit.app/api/verify \
  -H "X-API-Key: YOUR_KEY" \
  -F "files=@/path/to/doc1.pdf" \
  -F "files=@/path/to/doc2.png" \
  -F "files=@/path/to/doc3.docx" \
  -F 'expectations=["document is a signed contract", "effective date is in 2025"]'
```

| Form field | Type | Required | Description |
|---|---|---|---|
| `files` | file(s) | Yes | One or more files (repeat the field for each file, max 20) |
| `expectations` | string (JSON) | Yes | A JSON array of expectation strings, or a single JSON string |
| `model` | string | No | LLM model override |

### Response — Single File (200 JSON)

When only one file is submitted, the response is a plain JSON object identical to the `/api/verify/single` response.

### Response — Multiple Files (200 SSE Stream)

When two or more files are submitted, results are streamed as **Server-Sent Events**. Each event contains a `data:` line with a JSON object.

**Per-file event:**

```
data: {"index": 0, "filename": "doc1.pdf", "status": "ok", "matches": [...], "summary": "All expectations met."}

data: {"index": 1, "filename": "doc2.png", "status": "not_ok", "matches": [...], "summary": "Document is not a signed contract."}

data: {"index": 2, "filename": "doc3.docx", "status": "ok", "matches": [...], "summary": "All expectations met."}
```

Each per-file event includes an `index` field (0-based) corresponding to the upload order.

**Final summary event:**

After all files have been processed, a final event is emitted:

```
data: {"done": true, "total": 3, "ok_count": 2, "not_ok_count": 1}
```

| Field | Type | Description |
|---|---|---|
| `done` | boolean | Always `true` — signals the stream is complete |
| `total` | integer | Total number of files processed |
| `ok_count` | integer | Number of files where all expectations passed |
| `not_ok_count` | integer | Number of files where at least one expectation failed |

### Consuming the SSE Stream

**cURL (watch events in the terminal):**

```bash
curl -N -H "X-API-Key: YOUR_KEY" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.png" \
  -F 'expectations=["is a valid ID document"]' \
  https://idpai.replit.app/api/verify
```

**Python (using `httpx`):**

```python
import httpx
import json

files = [
    ("files", ("doc1.pdf", open("doc1.pdf", "rb"), "application/pdf")),
    ("files", ("doc2.png", open("doc2.png", "rb"), "image/png")),
]
data = {"expectations": '["is a valid ID document"]'}

with httpx.stream(
    "POST",
    "https://idpai.replit.app/api/verify",
    headers={"X-API-Key": "YOUR_KEY"},
    files=files,
    data=data,
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("done"):
                print(f"Complete: {event['ok_count']}/{event['total']} passed")
            else:
                print(f"[{event['index']}] {event['filename']}: {event['status']}")
```

> **Note:** Browser-based `EventSource` does not support custom headers or POST requests. Use `fetch()` with `ReadableStream` or a library like `eventsource-parser` for browser clients.

---

## 5. Response Field Reference

### Top-level result object

| Field | Type | Description |
|---|---|---|
| `filename` | string | Name of the uploaded file |
| `status` | string | `"ok"` if all expectations pass, `"not_ok"` otherwise |
| `matches` | array | Per-expectation results (see below) |
| `summary` | string | One-sentence overall assessment |

### Each `matches[]` entry

| Field | Type | Values | Description |
|---|---|---|---|
| `expected` | string | — | The expectation text as submitted |
| `result` | string | `"ok"`, `"not_ok"` | Whether the expectation was satisfied |
| `confidence` | string | `"high"`, `"medium"`, `"low"` | LLM confidence level |
| `details` | string | — | Explanation of why the expectation passed or failed |

---

## 6. File Constraints

| Constraint | Value |
|---|---|
| **Max file size** | 20 MB per file |
| **Max files per batch** | 20 |
| **Supported formats** | PDF, PNG, JPG/JPEG, TIFF/TIF, BMP, WebP, GIF, DOCX, DOC, TXT, MD, HTML/HTM, CSV |

**How files are processed:**

| Format | Processing method |
|---|---|
| Images (PNG, JPG, etc.) | Sent directly to the vision model as base64 |
| PDF | First 5 pages rendered as images + text extracted; both sent to the model |
| DOCX/DOC | Text extracted from paragraphs |
| TXT, MD, CSV | Raw text (truncated to 20,000 characters) |
| HTML/HTM | Parsed to plain text via BeautifulSoup |

---

## 7. Error Handling

### HTTP error responses

| Status | Cause | Example `detail` |
|---|---|---|
| `400` | Unsupported file format | `"Unsupported file type '.xyz'. Supported: .bmp, .csv, .doc, ..."` |
| `400` | File exceeds 20 MB | `"File too large. Maximum size is 20MB."` |
| `400` | Individual file too large (batch) | `"File 'report.pdf' too large. Maximum size is 20MB."` |
| `400` | Too many files | `"Too many files. Maximum is 20."` |
| `400` | Invalid expectations JSON | `"Invalid JSON in 'expectations' field."` |
| `400` | Wrong expectations type | `"'expectations' must be a JSON string or array of strings."` |
| `400` | Empty expectations | `"At least one expectation is required."` |
| `401` | Missing or invalid auth | `"Not authenticated"` |

### Graceful failures within SSE

When a multi-file batch encounters an error processing a specific file, the stream does **not** terminate. Instead, that file's event is emitted with `status: "not_ok"` and each match entry contains the error message in `details`. The final summary event still includes the file in `not_ok_count`.

---

## 8. Integration Tips

1. **Write clear, specific expectations.** Instead of "is valid", write "the document is a signed employment contract dated in 2025".

2. **Use the `model` parameter for complex documents.** The default model (`gpt-4o`) works well for most cases, but you can specify a different model if your deployment supports it.

3. **Batch when possible.** The multi-file endpoint processes documents sequentially and streams results. For large batches, submit up to 20 files per request and consume events as they arrive.

4. **Handle `"low"` confidence results.** Flag these for human review rather than making automated decisions based on them.

5. **Privacy-safe by design.** The verifier checks document *type and structure* for identity documents (Aadhaar, PAN, passports) without extracting or exposing personal data.

6. **Integrate with the document pipeline.** For documents already uploaded to IDP Kit, you can download them via `GET /api/documents/{doc_id}/download` and pipe the bytes to the verify endpoint, or upload fresh files directly.
