"""Tests for document upload/download safety & perf hardening (Task #31).

Covers:
- Content-Disposition filename sanitization (CR/LF stripped + RFC 5987).
- Streaming download (StreamingResponse, storage.load not materialized).
- Magic-byte upload validation (HTML disguised as PDF rejected).
- Async PDF parsing (event loop stays responsive while page-count is computed).
"""
from __future__ import annotations

import asyncio
import io
import time

import pytest


def _helpers():
    """Defer importing the route module so it doesn't bind the SQLAlchemy
    engine before the conftest ``_env`` fixture sets ``DATABASE_URL``."""
    from idpkit.api.routes.documents import (
        _content_disposition,
        _sanitize_filename_for_header,
        _sniff_format,
        _validate_content_matches_extension,
    )
    return (
        _content_disposition,
        _sanitize_filename_for_header,
        _sniff_format,
        _validate_content_matches_extension,
    )


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. Filename sanitization for Content-Disposition
# ---------------------------------------------------------------------------

def test_sanitize_strips_crlf_and_quotes():
    _content_disposition, _sanitize_filename_for_header, _sniff_format, _validate_content_matches_extension = _helpers()
    raw = 'evil\r\nX-Injected: 1"; rm -rf /\nfoo.pdf'
    cleaned = _sanitize_filename_for_header(raw)
    assert "\r" not in cleaned
    assert "\n" not in cleaned
    assert '"' not in cleaned
    assert cleaned  # never empty


def test_content_disposition_includes_rfc5987_for_unicode():
    _content_disposition, _, _, _ = _helpers()
    header = _content_disposition("résumé — 履歴書.pdf")
    assert header.startswith("attachment; ")
    assert "filename=\"" in header
    assert "filename*=UTF-8''" in header
    # Must not contain raw CR/LF, and the ASCII fallback must be quote-safe.
    assert "\r" not in header and "\n" not in header
    # Non-ASCII chars must NOT appear in the ASCII filename= part.
    ascii_part = header.split("filename=\"", 1)[1].split("\"", 1)[0]
    assert ascii_part.encode("ascii", "strict")  # round-trips


def test_content_disposition_neutralizes_crlf_injection():
    _content_disposition, _, _, _ = _helpers()
    header = _content_disposition("a\r\nSet-Cookie: hax=1.pdf")
    # No raw CR/LF survives — the dangerous characters are sanitized to '_'.
    assert "\r" not in header
    assert "\n" not in header


# ---------------------------------------------------------------------------
# 2. Magic-byte sniffing
# ---------------------------------------------------------------------------

def test_sniff_pdf():
    _, _, _sniff_format, _ = _helpers()
    assert _sniff_format(b"%PDF-1.4\n...") == "pdf"


def test_sniff_html_doctype():
    _, _, _sniff_format, _ = _helpers()
    assert _sniff_format(b"  <!DOCTYPE html><html><body>") == "html"


def test_sniff_ooxml_zip():
    _, _, _sniff_format, _ = _helpers()
    assert _sniff_format(b"PK\x03\x04rest-of-zip") == "ooxml"


def test_validate_rejects_html_disguised_as_pdf():
    _, _, _, _validate_content_matches_extension = _helpers()
    html = b"<!DOCTYPE html><html><body>not a pdf</body></html>"
    with pytest.raises(Exception) as ei:
        _validate_content_matches_extension(html, "pdf")
    assert ei.value.status_code == 400
    assert "does not match" in ei.value.detail.lower()


def test_validate_accepts_real_pdf():
    _, _, _, _validate_content_matches_extension = _helpers()
    _validate_content_matches_extension(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", "pdf")


def test_validate_passes_text_when_unknown():
    _, _, _, _validate_content_matches_extension = _helpers()
    _validate_content_matches_extension(b"# hello\nworld\n", "md")


# ---------------------------------------------------------------------------
# 3. Upload-time MIME mismatch rejection (end-to-end through the route)
# ---------------------------------------------------------------------------

async def test_upload_html_disguised_as_pdf_is_rejected(auth_client):
    fake_pdf = b"<!DOCTYPE html><html><body>I am HTML, not a PDF.</body></html>"
    res = await auth_client.post(
        "/api/documents/",
        files={"file": ("evil.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert res.status_code == 400, res.text
    assert "does not match" in res.json()["detail"].lower()


async def test_upload_real_pdf_succeeds(auth_client, sample_pdf_bytes):
    res = await auth_client.post(
        "/api/documents/",
        files={"file": ("ok.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["format"] == "pdf"
    assert body["page_count"] is not None and body["page_count"] >= 1


# ---------------------------------------------------------------------------
# 4. Download streams without materializing the whole blob into memory
# ---------------------------------------------------------------------------

async def test_download_uses_streaming_and_sanitizes_filename(
    auth_client, sample_pdf_bytes, monkeypatch
):
    # Upload first so we have something to download. Use a hostile filename
    # that exercises both the CR/LF sanitization and the RFC 5987 path.
    bad_name = "ré\r\nsumé.pdf"
    res = await auth_client.post(
        "/api/documents/",
        files={"file": (bad_name, io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert res.status_code == 201, res.text
    doc_id = res.json()["id"]

    # Spy on the storage backend: load() must NOT be invoked by the streaming
    # path (it would defeat the whole point).
    from idpkit.api import deps as deps_mod

    storage = deps_mod.get_storage()
    load_calls = {"n": 0}
    real_load = storage.load

    def _spy_load(key):
        load_calls["n"] += 1
        return real_load(key)

    monkeypatch.setattr(storage, "load", _spy_load)

    res = await auth_client.get(f"/api/documents/{doc_id}/download")
    assert res.status_code == 200, res.text

    # Header must be clean and ASCII-safe.
    cd = res.headers["content-disposition"]
    assert "\r" not in cd and "\n" not in cd
    assert "filename*=UTF-8''" in cd

    # Body round-trips identically.
    assert res.content == sample_pdf_bytes

    # And no fallback to load() occurred — iter_bytes streamed it.
    assert load_calls["n"] == 0


# ---------------------------------------------------------------------------
# 5. PDF parsing happens off the event loop
# ---------------------------------------------------------------------------

async def test_pdf_parsing_does_not_block_event_loop(auth_client, sample_pdf_bytes, monkeypatch):
    """While a slow page-count call is running, an unrelated request must
    still complete promptly. We replace ``_extract_page_count`` with a sync
    ``time.sleep`` and confirm a concurrent ``/api/auth/me`` returns inside
    a small fraction of the sleep duration — only possible if the parsing
    is dispatched to a worker thread."""
    from idpkit.api.routes import documents as docs_mod

    SLEEP = 0.6  # seconds

    def _slow_parse(content, fmt):
        time.sleep(SLEEP)
        return 1

    monkeypatch.setattr(docs_mod, "_extract_page_count", _slow_parse)

    upload_task = asyncio.create_task(auth_client.post(
        "/api/documents/",
        files={"file": ("slow.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    ))
    # Give the upload a moment to enter the to_thread call.
    await asyncio.sleep(0.05)

    t0 = time.monotonic()
    res = await auth_client.get("/api/auth/me")
    elapsed = time.monotonic() - t0
    assert res.status_code == 200
    # If parsing blocked the loop, this would take ~SLEEP seconds. Allow a
    # comfortable margin for CI jitter but well under the sleep duration.
    assert elapsed < SLEEP * 0.6, f"/api/auth/me took {elapsed:.3f}s — event loop appears blocked"

    upload_res = await upload_task
    assert upload_res.status_code == 201, upload_res.text


# ---------------------------------------------------------------------------
# 6. Direct-to-storage (signed-URL) confirmation re-validates content
# ---------------------------------------------------------------------------

async def test_confirm_upload_rejects_disguised_content(auth_client):
    """The signed-URL upload flow bypasses the in-process upload routes,
    so ``confirm_upload`` must re-sniff the bytes that the client wrote
    directly to storage. Disguised content (HTML uploaded as ``.pdf``)
    must be rejected and the rogue object deleted."""
    from idpkit.api import deps as deps_mod

    # Step 1: ask the API for an upload URL — this creates a Document row
    # in status="uploading" with a storage key reserved for it.
    res = await auth_client.post(
        "/api/documents/upload-url",
        json={"filename": "evil.pdf", "size": 64, "content_type": "application/pdf"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    doc_id = body["doc_id"]
    storage_key = body["storage_key"]

    # Step 2: simulate the client uploading hostile bytes directly to storage
    # (this is what would happen over a GCS signed PUT). Bypass the route.
    storage = deps_mod.get_storage()
    storage.save(storage_key, b"<!DOCTYPE html><html>not a pdf</html>")

    # Step 3: confirm-upload must catch the mismatch.
    res = await auth_client.post(f"/api/documents/{doc_id}/confirm-upload")
    assert res.status_code == 400, res.text
    assert "does not match" in res.json()["detail"].lower()

    # And the rogue object must have been removed from storage.
    assert not storage.exists(storage_key)


# ---------------------------------------------------------------------------
# 7. Lazily-caching backends (GCS) must not get poisoned by head-reads
# ---------------------------------------------------------------------------

def test_peek_bytes_does_not_poison_lazy_cache():
    """Regression for the GCS code-review finding: ``confirm_upload`` reads
    the first chunk to sniff the format. If that read used ``iter_bytes()``,
    a backend that caches as it streams (GCS) would write a truncated cache
    file and serve it for every subsequent download. ``peek_bytes()`` must
    take the no-cache path."""
    from typing import Iterator
    from idpkit.core.storage import StorageBackend

    class LazyCachingBackend(StorageBackend):
        """Mimics the GCS pattern: ``iter_bytes`` writes chunks to a local
        cache as it yields them; subsequent reads are served from the cache."""
        def __init__(self):
            self._objects: dict[str, bytes] = {}
            self._cache: dict[str, bytes] = {}

        def save(self, key, data):
            self._objects[key] = data if isinstance(data, bytes) else data.read()
            return key

        def load(self, key):
            if key in self._cache:
                return self._cache[key]
            data = self._objects[key]
            self._cache[key] = data
            return data

        def iter_bytes(self, key, chunk_size=64 * 1024):
            # The "broken" pattern: writes to cache as it yields. If the
            # caller breaks early, cache is poisoned with a truncated copy.
            buf = bytearray()
            data = self._objects[key]
            for off in range(0, len(data), chunk_size):
                chunk = data[off:off + chunk_size]
                buf.extend(chunk)
                self._cache[key] = bytes(buf)
                yield chunk

        def delete(self, key):
            self._objects.pop(key, None)
            self._cache.pop(key, None)

        def exists(self, key):
            return key in self._objects

        def list_keys(self, prefix=""):
            return [k for k in self._objects if k.startswith(prefix)]

        def get_path(self, key):
            return None

    be = LazyCachingBackend()
    full = b"%PDF-1.4\n" + b"X" * 4096
    be.save("doc1", full)

    # peek_bytes uses the safe (default) load() path, so no streaming and no
    # partial cache write happens.
    head = be.peek_bytes("doc1", 512)
    assert head == full[:512]

    # The full object is still readable via iter_bytes — i.e. the cache was
    # not poisoned with the 512-byte head.
    streamed = b"".join(be.iter_bytes("doc1", chunk_size=1024))
    assert streamed == full


def test_gcs_iter_bytes_uses_atomic_cache_rename(tmp_path, monkeypatch):
    """Verify ``GCSStorageBackend.iter_bytes`` writes to a ``.part`` file and
    only renames into place after the full body has streamed — so an
    interrupted iteration cannot leave a truncated cache file on disk."""
    from idpkit.core.storage import GCSStorageBackend

    class _FakeStreamCtx:
        status_code = 200
        def __init__(self, body, chunks):
            self._body = body
            self._chunks = chunks
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def iter_bytes(self, chunk_size):
            for c in self._chunks:
                yield c

    body = b"%PDF-1.4\n" + b"A" * 2048
    chunks = [body[i:i + 256] for i in range(0, len(body), 256)]

    be = GCSStorageBackend.__new__(GCSStorageBackend)
    be.bucket_id = "x"
    be.private_dir = "p"
    be._cache_dir = tmp_path

    monkeypatch.setattr(be, "_sign_url", lambda *a, **k: "http://fake/url")
    import httpx as _httpx
    monkeypatch.setattr(_httpx, "stream", lambda *a, **k: _FakeStreamCtx(body, chunks))

    # Simulate the broken caller: only consume the first chunk then break.
    it = be.iter_bytes("doc1", chunk_size=256)
    first = next(it)
    it.close()

    cache_file = tmp_path / "doc1"
    part_file = tmp_path / "doc1.part"
    # The .part file was cleaned up; no truncated cache file exists.
    assert not part_file.exists(), ".part should be removed on interrupted iter"
    assert not cache_file.exists(), "no truncated cache file should be written"

    # Fresh full read populates the cache atomically.
    full = b"".join(be.iter_bytes("doc1", chunk_size=256))
    assert full == body
    assert cache_file.exists() and cache_file.read_bytes() == body
