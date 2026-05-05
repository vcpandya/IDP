"""IDP Kit Document API routes — upload, list, get, delete, download."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote as _urlquote

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from fastapi.responses import Response as RawResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.db.session import get_db
from idpkit.db.models import Document, User
from idpkit.api.deps import get_current_user, get_storage, get_llm
from idpkit.core.storage import StorageBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Supported file formats mapped from extension -> canonical format name
EXTENSION_FORMAT_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".pptx": "pptx",
    ".ppt": "pptx",
    # Images
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
}

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# MIME type hints (used for download Content-Type)
FORMAT_CONTENT_TYPE: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "md": "text/markdown",
    "html": "text/html",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image": "application/octet-stream",  # overridden per-file when possible
}

IMAGE_CONTENT_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TagBrief(BaseModel):
    id: str
    name: str
    color: str

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: str
    filename: str
    format: Optional[str] = None
    file_size: int = 0
    page_count: Optional[int] = None
    total_tokens: Optional[int] = None
    status: str = "uploaded"
    description: Optional[str] = None
    metadata_json: Optional[dict] = None
    tree_index: Optional[dict] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    owner_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: list[TagBrief] = []

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    skip: int
    limit: int


class MessageResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_format(filename: str) -> tuple[str, str]:
    """Return (format, extension) from the filename.

    Raises HTTPException 400 if the extension is not supported.
    """
    import os
    ext = os.path.splitext(filename)[1].lower()
    fmt = EXTENSION_FORMAT_MAP.get(ext)
    if not fmt:
        supported = ", ".join(sorted(set(EXTENSION_FORMAT_MAP.values())))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {supported}",
        )
    return fmt, ext


def _storage_key(user_id: str, doc_id: str, ext: str) -> str:
    """Build the canonical storage key for an original upload."""
    return f"{user_id}/{doc_id}/original{ext}"


def _sniff_format(head: bytes) -> Optional[str]:
    """Best-effort content-type detection from the first ~512 bytes.

    Returns one of the canonical format names used in ``EXTENSION_FORMAT_MAP``,
    or ``None`` if the content isn't confidently identifiable. ``None`` means
    "let the declared extension stand" — used for plain-text formats (md, csv,
    html when no DOCTYPE) where there is no reliable magic.
    """
    if not head:
        return None
    if head.startswith(b"%PDF-"):
        return "pdf"
    # ZIP container — covers DOCX, XLSX, PPTX (all OOXML).
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "ooxml"
    # Legacy OLE compound (old .doc/.xls/.ppt).
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"
    # Image magic numbers.
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image"
    if head.startswith(b"BM"):
        return "image"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    # HTML — sniff a doctype or root tag in the first chunk, case-insensitive.
    snippet = head[:512].lstrip().lower()
    if snippet.startswith(b"<!doctype html") or snippet.startswith(b"<html") or b"<html" in snippet[:256]:
        return "html"
    return None


# Maps declared canonical format -> set of sniff results that are acceptable
# matches. Anything else for a declared extension is rejected.
_FORMAT_SNIFF_ALLOW: dict[str, set[Optional[str]]] = {
    "pdf": {"pdf"},
    "docx": {"ooxml", "ole"},
    "xlsx": {"ooxml", "ole"},
    "pptx": {"ooxml", "ole"},
    "image": {"image"},
    # Text-like formats: sniffing is best-effort; allow None (unknown), but
    # reject if it sniffs as an obviously different binary container.
    "md": {None},
    "html": {"html", None},
    "csv": {None},
}


def _validate_content_matches_extension(content: bytes, fmt: str) -> None:
    """Raise HTTPException 400 if the file's bytes obviously contradict the
    declared extension (e.g. an HTML file uploaded as ``.pdf``)."""
    sniffed = _sniff_format(content[:512])
    allowed = _FORMAT_SNIFF_ALLOW.get(fmt)
    if allowed is None:
        return  # No expectation registered for this format.
    if sniffed in allowed:
        return
    if sniffed is None and None in allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"File content does not match extension. Declared format '{fmt}' "
            f"but content sniffed as '{sniffed or 'unknown'}'."
        ),
    )


_UNSAFE_FILENAME_CHARS = re.compile(r'[\r\n\x00-\x1f"\\/]')


def _sanitize_filename_for_header(name: str) -> str:
    """Strip characters unsafe in an HTTP header (CR/LF, NUL, control chars,
    backslash, quote, slash). Returns a non-empty ASCII-safe display name; the
    full UTF-8 name is separately exposed via the RFC 5987 ``filename*`` field
    in ``_content_disposition``."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", name or "").strip()
    return cleaned or "download"


def _content_disposition(filename: str) -> str:
    """Build a CR/LF-safe ``Content-Disposition`` value with both an ASCII
    fallback (``filename=``) and a UTF-8-aware ``filename*=`` form per RFC 5987
    so emojis/accents/non-Latin names render correctly."""
    safe = _sanitize_filename_for_header(filename)
    ascii_fallback = safe.encode("ascii", "replace").decode("ascii").replace("?", "_")
    encoded = _urlquote(safe.encode("utf-8"), safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"


def _extract_page_count(content: bytes, fmt: str) -> Optional[int]:
    """Best-effort page count extraction from file bytes."""
    if fmt == "pdf":
        try:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except Exception:
            pass
    elif fmt == "pptx":
        try:
            import io
            from pptx import Presentation
            prs = Presentation(io.BytesIO(content))
            return len(prs.slides)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Upload a document file.

    Supported formats: PDF, DOCX, MD, HTML, XLSX, CSV, PPTX, and common
    image types (PNG, JPG, GIF, BMP, TIFF, WEBP, SVG).

    The file is saved to storage under ``{user_id}/{doc_id}/original.{ext}``
    and a ``Document`` database record is created with status ``uploaded``.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    fmt, ext = _detect_format(file.filename)

    # Create DB record first so we have the doc_id
    doc = Document(
        filename=file.filename,
        format=fmt,
        owner_id=user.id,
        status="uploaded",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Read file content with size check
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(content)} bytes). Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    # Reject polyglot / disguised uploads where bytes contradict the extension.
    _validate_content_matches_extension(content, fmt)

    key = _storage_key(user.id, doc.id, ext)

    try:
        storage.save(key, content)
    except Exception as exc:
        logger.error("Storage write failed for document %s: %s", doc.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded file",
        ) from exc

    # Update document with storage metadata
    doc.file_path = key
    doc.file_size = len(content)

    # Extract page count for supported formats — off the event loop so other
    # requests stay responsive while parsing multi-MB PDFs.
    page_count = await asyncio.to_thread(_extract_page_count, content, fmt)
    if page_count is not None:
        doc.page_count = page_count

    db.add(doc)
    await db.flush()

    result = await db.execute(
        select(Document)
        .options(selectinload(Document.tags))
        .where(Document.id == doc.id)
    )
    doc = result.scalar_one()

    logger.info(
        "Document uploaded: %s (id=%s, format=%s, size=%d, pages=%s)",
        doc.filename, doc.id, doc.format, doc.file_size, doc.page_count,
    )
    return doc


class UploadUrlRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str = Field(default="application/octet-stream")
    size: int = Field(..., gt=0)


class UploadUrlResponse(BaseModel):
    upload_url: str
    doc_id: str
    storage_key: str
    uses_signed_url: bool


@router.post(
    "/upload-url",
    response_model=UploadUrlResponse,
    summary="Get a signed upload URL for direct-to-storage upload",
)
async def get_upload_url(
    body: UploadUrlRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    if body.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    fmt, ext = _detect_format(body.filename)

    doc = Document(
        filename=body.filename,
        format=fmt,
        owner_id=user.id,
        status="uploading",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    key = _storage_key(user.id, doc.id, ext)
    doc.file_path = key
    doc.file_size = body.size
    db.add(doc)
    await db.flush()

    if storage.supports_signed_urls:
        signed_url = storage.get_signed_upload_url(key, body.content_type)
        logger.info("Signed upload URL generated for doc %s (%s)", doc.id, body.filename)
        return UploadUrlResponse(
            upload_url=signed_url,
            doc_id=doc.id,
            storage_key=key,
            uses_signed_url=True,
        )
    else:
        return UploadUrlResponse(
            upload_url=f"/api/documents/{doc.id}/upload-content",
            doc_id=doc.id,
            storage_key=key,
            uses_signed_url=False,
        )


@router.post(
    "/{doc_id}/upload-content",
    response_model=DocumentResponse,
    summary="Upload file content for a pre-created document (local storage fallback)",
)
async def upload_content(
    doc_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "uploading":
        raise HTTPException(status_code=400, detail="Document is not awaiting upload")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    _validate_content_matches_extension(content, doc.format or "")

    try:
        storage.save(doc.file_path, content)
    except Exception as exc:
        logger.error("Storage write failed for document %s: %s", doc.id, exc)
        raise HTTPException(status_code=500, detail="Failed to store uploaded file")

    doc.file_size = len(content)
    doc.status = "uploaded"
    page_count = await asyncio.to_thread(_extract_page_count, content, doc.format)
    if page_count is not None:
        doc.page_count = page_count
    db.add(doc)
    await db.flush()

    result = await db.execute(
        select(Document)
        .options(selectinload(Document.tags))
        .where(Document.id == doc.id)
    )
    doc = result.scalar_one()
    return doc


@router.post(
    "/{doc_id}/confirm-upload",
    response_model=DocumentResponse,
    summary="Confirm a direct-to-storage upload completed",
)
async def confirm_upload(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "uploading":
        raise HTTPException(status_code=400, detail="Document is not awaiting upload confirmation")

    if not storage.exists(doc.file_path):
        raise HTTPException(status_code=400, detail="File not found in storage. Upload may have failed.")

    # Direct-to-storage uploads bypass the in-process upload routes, so the
    # MIME magic-byte check that protects upload_document/upload_content was
    # never run. Re-validate here against the first chunk; if it fails, delete
    # the rogue object so it can't be served or referenced later.
    try:
        head = b""
        for chunk in storage.iter_bytes(doc.file_path, chunk_size=512):
            head = chunk
            break
    except Exception as exc:  # pragma: no cover - storage backends differ
        logger.warning("confirm_upload: could not read head bytes for %s: %s", doc.id, exc)
        head = b""
    if head:
        try:
            _validate_content_matches_extension(head, doc.format or "")
        except HTTPException:
            try:
                storage.delete(doc.file_path)
            except Exception as del_exc:  # pragma: no cover
                logger.warning("confirm_upload: failed to delete rogue object %s: %s", doc.file_path, del_exc)
            await db.delete(doc)
            await db.flush()
            raise

    doc.status = "uploaded"
    db.add(doc)
    await db.flush()

    result = await db.execute(
        select(Document)
        .options(selectinload(Document.tags))
        .where(Document.id == doc.id)
    )
    doc = result.scalar_one()

    logger.info("Direct upload confirmed for doc %s (%s)", doc.id, doc.filename)
    return doc


@router.get(
    "/upload-mode",
    summary="Check whether direct upload is available",
)
async def get_upload_mode(
    storage: StorageBackend = Depends(get_storage),
):
    return {"direct_upload": storage.supports_signed_urls}


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List user's documents",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=500, description="Max records to return"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's documents with pagination."""
    base = select(Document).where(Document.owner_id == user.id)

    # Total count
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    # Paginated rows (eagerly load tags for response serialization)
    stmt = (
        base.options(selectinload(Document.tags))
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    docs = result.scalars().all()

    return DocumentListResponse(items=docs, total=total, skip=skip, limit=limit)


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document details",
)
async def get_document(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return full document details including ``tree_index`` if available."""
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.tags))
        .where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc


@router.delete(
    "/{doc_id}",
    response_model=MessageResponse,
    summary="Delete a document",
)
async def delete_document(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Delete a document record and its associated storage files.

    Removes the entire ``{user_id}/{doc_id}/`` directory from storage and
    deletes the database record (cascading to related jobs).
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Remove storage directory for this document
    storage_prefix = f"{user.id}/{doc_id}"
    try:
        if storage.exists(storage_prefix):
            storage.delete(storage_prefix)
    except Exception as exc:
        logger.warning("Storage cleanup failed for document %s: %s", doc_id, exc)
        # Continue with DB deletion even if storage cleanup fails

    await db.delete(doc)
    await db.flush()

    logger.info("Document deleted: %s (id=%s)", doc.filename, doc.id)
    return MessageResponse(detail=f"Document '{doc.filename}' deleted")


@router.get(
    "/{doc_id}/download",
    summary="Download the original document file",
    responses={
        200: {"description": "The document file", "content": {"application/octet-stream": {}}},
        404: {"description": "Document not found"},
    },
)
async def download_document(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Stream the original uploaded file back to the client."""
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if not doc.file_path or not storage.exists(doc.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found in storage",
        )

    # Determine content type
    import os
    ext = os.path.splitext(doc.filename)[1].lower()
    if doc.format == "image" and ext in IMAGE_CONTENT_TYPES:
        content_type = IMAGE_CONTENT_TYPES[ext]
    else:
        content_type = FORMAT_CONTENT_TYPE.get(doc.format or "", "application/octet-stream")

    headers = {"Content-Disposition": _content_disposition(doc.filename)}
    if doc.file_size:
        headers["Content-Length"] = str(doc.file_size)

    file_path = doc.file_path

    # Prime the stream by pulling the first chunk *before* returning the
    # response. If storage is broken or the object disappeared, this raises
    # cleanly into a 500/404 HTTPException instead of a half-sent response
    # body that the client would interpret as a truncated success.
    try:
        iterator = storage.iter_bytes(file_path)
        try:
            first_chunk = next(iterator)
        except StopIteration:
            first_chunk = b""
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found in storage",
        )
    except Exception as exc:
        logger.error("Storage stream failed to start for document %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read document from storage",
        )

    def _stream():
        if first_chunk:
            yield first_chunk
        try:
            yield from iterator
        except Exception as exc:
            # Response headers/status are already on the wire; we can no
            # longer convert this to an HTTP error. Log and abort the stream
            # so the client sees a truncated body rather than silent success.
            logger.error("Storage stream failed mid-transfer for document %s: %s", doc_id, exc)
            return

    return StreamingResponse(_stream(), media_type=content_type, headers=headers)


class AutoTagRequest(BaseModel):
    apply: bool = Field(False, description="If true, automatically apply suggested tags")


class AutoTagSuggestion(BaseModel):
    name: str
    existing_id: Optional[str] = None
    confidence: float


class AutoTagResponse(BaseModel):
    document_id: str
    suggestions: list[AutoTagSuggestion]
    applied: list[dict] = []


@router.post(
    "/{doc_id}/auto-tag",
    response_model=AutoTagResponse,
    summary="AI-powered auto-tagging for a document",
)
async def auto_tag_document(
    doc_id: str,
    body: Optional[AutoTagRequest] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body is None:
        body = AutoTagRequest()

    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from idpkit.engine.auto_tagger import suggest_tags, apply_tags
    from idpkit.core.llm import LLMClient

    llm = get_llm()
    suggestions = await suggest_tags(doc_id, user.id, db, llm)

    applied = []
    if body.apply and suggestions:
        applied = await apply_tags(doc_id, suggestions, user.id, db)

    return AutoTagResponse(
        document_id=doc_id,
        suggestions=[AutoTagSuggestion(**s) for s in suggestions],
        applied=applied,
    )
