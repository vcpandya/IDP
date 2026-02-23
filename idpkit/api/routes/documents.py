"""IDP Kit Document API routes — upload, list, get, delete, download."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from fastapi.responses import Response as RawResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import Document, User
from idpkit.api.deps import get_current_user, get_storage
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
    ".svg": "image",
}

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
    ".svg": "image/svg+xml",
}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

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
    owner_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

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

    # Read file content and store
    content = await file.read()
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
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    logger.info(
        "Document uploaded: %s (id=%s, format=%s, size=%d)",
        doc.filename, doc.id, doc.format, doc.file_size,
    )
    return doc


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List user's documents",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's documents with pagination."""
    base = select(Document).where(Document.owner_id == user.id)

    # Total count
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    # Paginated rows
    stmt = base.order_by(Document.created_at.desc()).offset(skip).limit(limit)
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
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
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

    try:
        data = storage.load(doc.file_path)
    except Exception as exc:
        logger.error("Storage read failed for document %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read document from storage",
        ) from exc

    # Determine content type
    import os
    ext = os.path.splitext(doc.filename)[1].lower()
    if doc.format == "image" and ext in IMAGE_CONTENT_TYPES:
        content_type = IMAGE_CONTENT_TYPES[ext]
    else:
        content_type = FORMAT_CONTENT_TYPE.get(doc.format or "", "application/octet-stream")

    return RawResponse(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{doc.filename}"',
            "Content-Length": str(len(data)),
        },
    )
