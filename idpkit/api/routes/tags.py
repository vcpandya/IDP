"""IDP Kit Tags API routes — CRUD for document tags / knowledge-base groups."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.db.session import get_db, lock_tag_name
from idpkit.db.models import (
    Document,
    Tag,
    User,
    conversation_tags,
    document_tags,
    generate_uuid,
)
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


class TagMergeRequest(BaseModel):
    source_tag_ids: list[str] = Field(..., min_length=1)


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
    """Create a new tag, reusing an existing same-name tag if one exists.

    Tags act as folders, so a duplicate name would silently fork a "folder".
    We look the name up case-insensitively and return the existing tag instead
    of creating a second one (idempotent create).
    """
    name = body.name.strip()
    await lock_tag_name(db, user.id, name)
    existing = (
        await db.execute(
            select(Tag).where(
                Tag.owner_id == user.id,
                func.lower(Tag.name) == name.lower(),
            )
        )
    ).scalars().first()
    if existing is not None:
        count = (
            await db.execute(
                select(func.count())
                .select_from(document_tags)
                .where(document_tags.c.tag_id == existing.id)
            )
        ).scalar() or 0
        return TagResponse(
            id=existing.id,
            name=existing.name,
            color=existing.color,
            description=existing.description,
            document_count=count,
            created_at=existing.created_at.isoformat() if existing.created_at else None,
            updated_at=existing.updated_at.isoformat() if existing.updated_at else None,
        )

    tag = Tag(
        id=generate_uuid(),
        name=name,
        color=body.color,
        description=body.description,
        owner_id=user.id,
    )
    db.add(tag)
    # The advisory lock above serialized the existence check + insert per
    # (owner, lower(name)), so a concurrent same-name create cannot have slipped
    # in between — a plain flush is safe here.
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
        .limit(200)
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
        new_name = body.name.strip()
        if new_name and new_name.lower() != tag.name.lower():
            # Serialize against concurrent create/rename to the same name so the
            # clash check + rename can't race another writer into a duplicate.
            await lock_tag_name(db, user.id, new_name)
            clash = (
                await db.execute(
                    select(Tag).where(
                        Tag.owner_id == user.id,
                        Tag.id != tag.id,
                        func.lower(Tag.name) == new_name.lower(),
                    )
                )
            ).scalars().first()
            if clash is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"A tag named '{clash.name}' already exists. "
                        "Merge them instead of renaming."
                    ),
                )
        if new_name:
            tag.name = new_name
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


@router.post("/{tag_id}/merge", response_model=MessageResponse)
async def merge_tags(
    tag_id: str,
    body: TagMergeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Merge one or more *source* tags into the *target* (``tag_id``).

    Every document (and conversation) linked to a source is relinked to the
    target without creating duplicate links, then the source tags are deleted.
    Used to consolidate accidental duplicate "folders".
    """
    target = (
        await db.execute(
            select(Tag).where(Tag.id == tag_id, Tag.owner_id == user.id)
        )
    ).scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Target tag not found")

    source_ids = [sid for sid in dict.fromkeys(body.source_tag_ids) if sid != tag_id]
    if not source_ids:
        raise HTTPException(status_code=400, detail="No source tags to merge")

    sources = (
        await db.execute(
            select(Tag).where(
                Tag.id.in_(source_ids), Tag.owner_id == user.id
            )
        )
    ).scalars().all()
    if not sources:
        raise HTTPException(status_code=404, detail="No matching source tags found")

    moved_docs = 0
    for src in sources:
        # Relink documents to the target. Conflict-safe insert avoids PK
        # collisions if a (document_id, target) link already exists or another
        # writer races us, then drop the source links.
        src_doc_ids = [
            row[0]
            for row in (
                await db.execute(
                    select(document_tags.c.document_id).where(
                        document_tags.c.tag_id == src.id
                    )
                )
            ).all()
        ]
        if src_doc_ids:
            res = await db.execute(
                pg_insert(document_tags)
                .values(
                    [
                        {"document_id": did, "tag_id": target.id}
                        for did in src_doc_ids
                    ]
                )
                .on_conflict_do_nothing()
            )
            moved_docs += res.rowcount or 0
        await db.execute(
            document_tags.delete().where(document_tags.c.tag_id == src.id)
        )

        # Same for conversation links.
        src_conv_ids = [
            row[0]
            for row in (
                await db.execute(
                    select(conversation_tags.c.conversation_id).where(
                        conversation_tags.c.tag_id == src.id
                    )
                )
            ).all()
        ]
        if src_conv_ids:
            await db.execute(
                pg_insert(conversation_tags)
                .values(
                    [
                        {"conversation_id": cid, "tag_id": target.id}
                        for cid in src_conv_ids
                    ]
                )
                .on_conflict_do_nothing()
            )
        await db.execute(
            conversation_tags.delete().where(conversation_tags.c.tag_id == src.id)
        )

        await db.delete(src)

    await db.flush()
    return MessageResponse(
        detail=(
            f"Merged {len(sources)} tag(s) into '{target.name}' "
            f"({moved_docs} document link(s) moved)"
        )
    )


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
