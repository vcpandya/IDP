"""IDP Kit Tags API routes — CRUD for document tags / knowledge-base groups."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.db.session import get_db
from idpkit.db.models import Document, Tag, User, document_tags, generate_uuid
from idpkit.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tags", tags=["tags"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TagCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    color: str = Field(default="#4f46e5", max_length=7)
    description: Optional[str] = Field(default=None, max_length=500)


class TagUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    color: Optional[str] = Field(default=None, max_length=7)
    description: Optional[str] = Field(default=None, max_length=500)


class TagDocumentsAdd(BaseModel):
    document_ids: list[str]


class TagDocumentInfo(BaseModel):
    id: str
    filename: str
    format: Optional[str] = None
    status: str = "uploaded"

    class Config:
        from_attributes = True


class TagResponse(BaseModel):
    id: str
    name: str
    color: str
    description: Optional[str] = None
    document_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TagDetailResponse(TagResponse):
    documents: list[TagDocumentInfo] = []


class MessageResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_tag(
    body: TagCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new tag."""
    tag = Tag(
        id=generate_uuid(),
        name=body.name,
        color=body.color,
        description=body.description,
        owner_id=user.id,
    )
    db.add(tag)
    await db.flush()
    await db.refresh(tag)
    return TagResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        description=tag.description,
        document_count=0,
        created_at=tag.created_at.isoformat() if tag.created_at else None,
        updated_at=tag.updated_at.isoformat() if tag.updated_at else None,
    )


@router.get("/")
async def list_tags(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List the current user's tags with document counts."""
    stmt = (
        select(Tag, func.count(document_tags.c.document_id).label("doc_count"))
        .outerjoin(document_tags, Tag.id == document_tags.c.tag_id)
        .where(Tag.owner_id == user.id)
        .group_by(Tag.id)
        .order_by(Tag.name)
    )
    rows = await db.execute(stmt)
    results = []
    for tag, doc_count in rows:
        results.append(TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            description=tag.description,
            document_count=doc_count,
            created_at=tag.created_at.isoformat() if tag.created_at else None,
            updated_at=tag.updated_at.isoformat() if tag.updated_at else None,
        ))
    return results


@router.get("/{tag_id}")
async def get_tag(
    tag_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get tag detail with its documents."""
    result = await db.execute(
        select(Tag)
        .options(selectinload(Tag.documents))
        .where(Tag.id == tag_id, Tag.owner_id == user.id)
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    return TagDetailResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        description=tag.description,
        document_count=len(tag.documents),
        created_at=tag.created_at.isoformat() if tag.created_at else None,
        updated_at=tag.updated_at.isoformat() if tag.updated_at else None,
        documents=[
            TagDocumentInfo(id=d.id, filename=d.filename, format=d.format, status=d.status)
            for d in tag.documents
        ],
    )


@router.patch("/{tag_id}")
async def update_tag(
    tag_id: str,
    body: TagUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a tag's name, color, or description."""
    result = await db.execute(
        select(Tag).where(Tag.id == tag_id, Tag.owner_id == user.id)
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    if body.name is not None:
        tag.name = body.name
    if body.color is not None:
        tag.color = body.color
    if body.description is not None:
        tag.description = body.description

    db.add(tag)
    await db.flush()
    await db.refresh(tag)

    # Get document count
    count_result = await db.execute(
        select(func.count()).select_from(document_tags).where(document_tags.c.tag_id == tag.id)
    )
    doc_count = count_result.scalar() or 0

    return TagResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        description=tag.description,
        document_count=doc_count,
        created_at=tag.created_at.isoformat() if tag.created_at else None,
        updated_at=tag.updated_at.isoformat() if tag.updated_at else None,
    )


@router.delete("/{tag_id}", response_model=MessageResponse)
async def delete_tag(
    tag_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a tag (documents are kept, only association removed)."""
    result = await db.execute(
        select(Tag).where(Tag.id == tag_id, Tag.owner_id == user.id)
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    await db.delete(tag)
    await db.flush()
    return MessageResponse(detail=f"Tag '{tag.name}' deleted")


@router.post("/{tag_id}/documents", response_model=MessageResponse)
async def add_documents_to_tag(
    tag_id: str,
    body: TagDocumentsAdd,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add documents to a tag."""
    result = await db.execute(
        select(Tag)
        .options(selectinload(Tag.documents))
        .where(Tag.id == tag_id, Tag.owner_id == user.id)
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    existing_ids = {d.id for d in tag.documents}
    new_ids = [did for did in body.document_ids if did not in existing_ids]

    if new_ids:
        docs_result = await db.execute(
            select(Document).where(
                Document.id.in_(new_ids),
                Document.owner_id == user.id,
            )
        )
        docs = docs_result.scalars().all()
        for doc in docs:
            tag.documents.append(doc)
        await db.flush()

    return MessageResponse(detail=f"Added {len(new_ids)} document(s) to tag '{tag.name}'")


@router.delete("/{tag_id}/documents/{doc_id}", response_model=MessageResponse)
async def remove_document_from_tag(
    tag_id: str,
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a document from a tag."""
    result = await db.execute(
        select(Tag)
        .options(selectinload(Tag.documents))
        .where(Tag.id == tag_id, Tag.owner_id == user.id)
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    tag.documents = [d for d in tag.documents if d.id != doc_id]
    await db.flush()
    return MessageResponse(detail="Document removed from tag")
