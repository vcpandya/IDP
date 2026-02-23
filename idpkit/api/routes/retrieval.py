"""IDP Kit Retrieval API routes — query documents using tree-based RAG."""

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
from idpkit.retrieval.tree_search import tree_search
from idpkit.retrieval.context_builder import build_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    model: Optional[str] = Field(None, description="LLM model override")
    max_context_tokens: Optional[int] = Field(
        4000, ge=100, le=32000, description="Max tokens for retrieved context"
    )


class SourceInfo(BaseModel):
    node_id: Optional[str] = None
    title: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    summary: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    query: str
    document_id: str


# ---------------------------------------------------------------------------
# Helper: build final answer from context + query
# ---------------------------------------------------------------------------

def _build_answer_prompt(query: str, context: str) -> str:
    """Build the prompt for generating a final answer from retrieved context."""
    return f"""You are a helpful document analysis assistant. Answer the user's question based ONLY on the provided document context. If the context does not contain enough information to answer the question, say so clearly.

When citing information, reference the section titles and page numbers.

{context}

User question: {query}

Provide a clear, well-structured answer based on the document context above."""


def _nodes_to_sources(nodes: list[dict]) -> list[SourceInfo]:
    """Convert raw tree search nodes into SourceInfo response objects."""
    sources = []
    for node in nodes:
        sources.append(
            SourceInfo(
                node_id=node.get("node_id"),
                title=node.get("title", "(untitled)"),
                start_page=node.get("start_index"),
                end_page=node.get("end_index"),
                summary=(node.get("summary") or node.get("prefix_summary") or "")[:500],
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/documents/{doc_id}/query",
    response_model=QueryResponse,
    summary="Query a single document using tree-based RAG",
)
async def query_document(
    doc_id: str,
    body: QueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Query a document using LLM-guided tree search and RAG.

    1. Retrieves the document and its tree index from the database.
    2. Runs tree_search to find relevant sections.
    3. Builds a context string from the relevant sections.
    4. Sends the context + query to the LLM for a final answer.
    """
    # 1. Fetch document and verify ownership
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if not doc.tree_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been indexed yet. Run indexing first.",
        )

    tree_idx = doc.tree_index
    if not tree_idx.get("structure"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document tree index has no structure. Re-index the document.",
        )

    # 2. Tree search — find relevant sections
    try:
        relevant_nodes = await tree_search(
            tree_index=tree_idx,
            query=body.query,
            llm=llm,
            model=body.model,
        )
    except Exception as exc:
        logger.error("Tree search failed for doc %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tree search failed: {exc}",
        )

    # 3. Build context from relevant nodes
    max_context = body.max_context_tokens or 4000
    context = build_context(relevant_nodes, max_tokens=max_context)

    if not context.strip():
        # No relevant sections found — still attempt an answer
        context = (
            "No highly relevant sections were found in the document. "
            "The document is titled: "
            + (tree_idx.get("doc_name") or doc.filename or "Unknown")
            + "."
        )
        if tree_idx.get("doc_description"):
            context += f" Description: {tree_idx['doc_description']}"

    # 4. Generate the final answer
    answer_prompt = _build_answer_prompt(body.query, context)
    try:
        llm_response = await llm.acomplete(answer_prompt, model=body.model)
        answer_text = llm_response.content
    except Exception as exc:
        logger.error("LLM answer generation failed for doc %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        )

    # 5. Build response
    sources = _nodes_to_sources(relevant_nodes)

    logger.info(
        "RAG query completed for doc %s — %d sources, answer length %d",
        doc_id,
        len(sources),
        len(answer_text),
    )

    return QueryResponse(
        answer=answer_text,
        sources=sources,
        query=body.query,
        document_id=doc_id,
    )
