"""IDP Kit Processing API routes — extraction, summarization, comparison."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import Document, User
from idpkit.api.deps import get_current_user, get_llm
from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/processing", tags=["processing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ExtractEntitiesRequest(BaseModel):
    document_id: str = Field(..., description="ID of the document to extract from")
    entity_types: Optional[list[str]] = Field(
        None, description="Entity types to extract (e.g. person, organization, date)"
    )


class EntityResponse(BaseModel):
    entity: str
    type: str
    context: Optional[str] = None


class ExtractEntitiesResponse(BaseModel):
    entities: list[EntityResponse]
    document_id: str
    count: int


class SummarizeRequest(BaseModel):
    document_id: str = Field(..., description="ID of the document to summarize")
    length: str = Field("standard", description="Summary length: 'brief', 'standard', or 'detailed'")


class SummarizeResponse(BaseModel):
    summary: str
    document_id: str
    length: str


class CompareRequest(BaseModel):
    document_id_1: str = Field(..., description="ID of the first document")
    document_id_2: str = Field(..., description="ID of the second document")


class CompareResponse(BaseModel):
    doc1_name: str
    doc2_name: str
    structural_comparison: str
    content_comparison: str
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_user_document(doc_id: str, user: User, db: AsyncSession) -> Document:
    """Fetch a document by ID, ensuring it belongs to the user."""
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id!r} not found",
        )
    return doc


def _get_document_text(doc: Document) -> str:
    """Extract text content from a document's tree index or parsed content."""
    if doc.tree_index:
        parts = []
        for node in doc.tree_index.get("structure", []):
            _collect_text(node, parts)
        if parts:
            return "\n\n".join(parts)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Document has not been indexed yet. Run indexing first.",
    )


def _collect_text(node: dict, parts: list[str]) -> None:
    """Recursively collect text from tree nodes."""
    text = node.get("text", "")
    summary = node.get("summary", "")
    title = node.get("title", "")

    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if text:
        parts.append(text)

    for child in node.get("nodes", []):
        _collect_text(child, parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/extract-entities",
    response_model=ExtractEntitiesResponse,
    summary="Extract named entities from a document",
)
async def extract_entities_route(
    body: ExtractEntitiesRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Extract named entities (people, organizations, dates, etc.) from a document."""
    doc = await _get_user_document(body.document_id, user, db)
    text = _get_document_text(doc)

    from idpkit.processing.extraction import extract_entities

    try:
        entities = await extract_entities(text, llm, entity_types=body.entity_types)
    except Exception as exc:
        logger.error("Entity extraction failed for doc %s: %s", body.document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {exc}",
        )

    return ExtractEntitiesResponse(
        entities=entities,
        document_id=body.document_id,
        count=len(entities),
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document",
)
async def summarize_route(
    body: SummarizeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Generate a summary of a document at the specified length."""
    doc = await _get_user_document(body.document_id, user, db)
    text = _get_document_text(doc)

    from idpkit.processing.summarization import summarize_text

    try:
        summary = await summarize_text(text, llm, length=body.length)
    except Exception as exc:
        logger.error("Summarization failed for doc %s: %s", body.document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {exc}",
        )

    return SummarizeResponse(
        summary=summary,
        document_id=body.document_id,
        length=body.length,
    )


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two documents",
)
async def compare_route(
    body: CompareRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Compare two documents structurally and by content."""
    doc1 = await _get_user_document(body.document_id_1, user, db)
    doc2 = await _get_user_document(body.document_id_2, user, db)

    if not doc1.tree_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document {body.document_id_1!r} has not been indexed.",
        )
    if not doc2.tree_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document {body.document_id_2!r} has not been indexed.",
        )

    from idpkit.processing.comparison import compare_documents

    try:
        result = await compare_documents(doc1.tree_index, doc2.tree_index, llm)
    except Exception as exc:
        logger.error("Document comparison failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {exc}",
        )

    return CompareResponse(**result)
