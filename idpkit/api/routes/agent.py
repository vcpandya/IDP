"""IDP Kit Agent API routes — conversational AI agent with tool-calling."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User, document_tags
from idpkit.api.deps import get_current_user, get_llm
from idpkit.core.llm import LLMClient
from idpkit.agent.agent import IDPAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the agent")
    document_ids: list[str] = Field(
        default_factory=list,
        description="List of document IDs in scope for this conversation",
    )
    tag_ids: list[str] = Field(
        default_factory=list,
        description="List of tag IDs — their documents are merged into document_ids",
    )


class ToolCallInfo(BaseModel):
    name: str
    args: dict
    result: Optional[dict] = None


class ChatSourceInfo(BaseModel):
    document_id: str
    filename: str
    node_id: Optional[str] = None
    title: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    summary: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    sources: list[ChatSourceInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sources(tool_call_log: list[dict]) -> list[ChatSourceInfo]:
    """Extract deduplicated source info from search_document tool results."""
    sources: list[ChatSourceInfo] = []
    seen: set[tuple[str, str]] = set()

    for tc in tool_call_log:
        if tc.get("name") != "search_document":
            continue
        result = tc.get("result") or {}
        doc_id = result.get("document_id", "")
        filename = result.get("filename", "")
        for node in result.get("results", []):
            nid = node.get("node_id", "")
            key = (doc_id, nid)
            if key in seen:
                continue
            seen.add(key)
            sources.append(ChatSourceInfo(
                document_id=doc_id,
                filename=filename,
                node_id=nid,
                title=node.get("title"),
                start_page=node.get("start_page"),
                end_page=node.get("end_page"),
                summary=(node.get("summary") or "")[:200],
            ))

    return sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the IDP Agent",
)
async def agent_chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Send a message to the IDP Agent and receive a response.

    The agent can call tools (search_document, list_documents,
    summarize_section, extract_data) to answer the user's question
    about their documents.
    """
    # Resolve tag_ids → document_ids and merge
    combined_doc_ids = list(body.document_ids)
    if body.tag_ids:
        stmt = select(document_tags.c.document_id).where(
            document_tags.c.tag_id.in_(body.tag_ids)
        )
        rows = await db.execute(stmt)
        tag_doc_ids = [r[0] for r in rows]
        for did in tag_doc_ids:
            if did not in combined_doc_ids:
                combined_doc_ids.append(did)

    agent = IDPAgent()

    try:
        result = await agent.chat(
            message=body.message,
            document_ids=combined_doc_ids,
            llm=llm,
            db=db,
        )
    except Exception as exc:
        logger.error("Agent chat failed for user %s: %s", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing failed: {exc}",
        )

    tool_calls_log = result.get("tool_calls", [])
    tool_calls = [
        ToolCallInfo(
            name=tc["name"],
            args=tc["args"],
            result=tc.get("result"),
        )
        for tc in tool_calls_log
    ]

    sources = _extract_sources(tool_calls_log)

    return ChatResponse(
        response=result["response"],
        tool_calls=tool_calls,
        sources=sources,
    )
