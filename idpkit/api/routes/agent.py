"""IDP Kit Agent API routes — conversational AI agent with tool-calling."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.db.session import get_db
from idpkit.db.models import (
    User, Document, document_tags, Conversation, ConversationMessage,
)
from idpkit.api.deps import get_current_user, get_llm
from idpkit.core.llm import LLMClient
from idpkit.agent.agent import IDPAgent
from idpkit.agent.memory import ConversationMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the agent")
    conversation_id: Optional[str] = Field(
        default=None, description="Existing conversation ID (omit to chat without history)",
    )
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


class SearchAttemptInfo(BaseModel):
    document_id: str
    filename: str
    query: str = ""
    results_found: int = 0
    status: str = "not_searched"  # found, no_results, error, not_searched


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    sources: list[ChatSourceInfo] = Field(default_factory=list)
    source_type: str = "general_knowledge"  # documents, general_knowledge, mixed
    search_attempts: list[SearchAttemptInfo] = Field(default_factory=list)


# -- Conversation CRUD schemas -----------------------------------------------

class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=200)


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ConversationMessageInfo(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    sources: Optional[list[ChatSourceInfo]] = None
    source_type: Optional[str] = None
    created_at: str


class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessageInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sources(tool_call_log: list[dict]) -> list[ChatSourceInfo]:
    """Extract deduplicated source info from tool results.

    Handles search_document, find_cross_references, and query_graph
    (entity_mentions operation).
    """
    sources: list[ChatSourceInfo] = []
    seen: set[tuple[str, str]] = set()

    def _add(doc_id: str, filename: str, node_id: str, title: str | None,
             start_page: int | None, end_page: int | None, summary: str | None = None):
        key = (doc_id, node_id or "")
        if key in seen or not doc_id:
            return
        seen.add(key)
        sources.append(ChatSourceInfo(
            document_id=doc_id,
            filename=filename,
            node_id=node_id or None,
            title=title,
            start_page=start_page,
            end_page=end_page,
            summary=(summary or "")[:200] or None,
        ))

    for tc in tool_call_log:
        name = tc.get("name", "")
        result = tc.get("result") or {}

        if name == "search_document":
            doc_id = result.get("document_id", "")
            filename = result.get("filename", "")
            for node in result.get("results", []):
                _add(doc_id, filename, node.get("node_id", ""),
                     node.get("title"), node.get("start_page"),
                     node.get("end_page"), node.get("summary"))

        elif name == "find_cross_references":
            for ref in result.get("references", []):
                _add(ref.get("document_id", ""),
                     ref.get("document_filename", ""),
                     ref.get("node_id", ""),
                     ref.get("node_title"),
                     ref.get("start_page"),
                     ref.get("end_page"),
                     ref.get("entity_name"))

        elif name == "query_graph":
            op = result.get("operation", "")
            if op == "entity_mentions":
                for m in result.get("mentions", []):
                    _add(m.get("document_id", ""), "",
                         m.get("node_id", ""),
                         m.get("node_title"),
                         m.get("start_page"),
                         m.get("end_page"),
                         m.get("mention_text"))

    return sources


def _sources_to_json(sources: list[ChatSourceInfo]) -> list[dict] | None:
    """Serialize sources to plain dicts for DB JSON column."""
    if not sources:
        return None
    return [s.model_dump(exclude_none=True) for s in sources]


def _sources_from_json(data) -> list[ChatSourceInfo]:
    """Deserialize sources from DB JSON column."""
    if not data:
        return []
    return [ChatSourceInfo(**s) for s in data]


def _classify_source_type(
    tool_call_log: list[dict], requested_doc_ids: list[str],
) -> str:
    """Classify whether the response is based on documents, general knowledge, or mixed."""
    if not requested_doc_ids:
        return "general_knowledge"

    search_doc_ids: set[str] = set()
    found_doc_ids: set[str] = set()

    for tc in tool_call_log:
        if tc.get("name") == "search_document":
            doc_id = tc.get("args", {}).get("document_id", "")
            if doc_id:
                search_doc_ids.add(doc_id)
            result = tc.get("result") or {}
            results = result.get("results", [])
            if results and doc_id:
                found_doc_ids.add(doc_id)

    if not search_doc_ids:
        return "general_knowledge"
    if found_doc_ids:
        if found_doc_ids == search_doc_ids:
            return "documents"
        return "mixed"
    return "general_knowledge"


def _extract_search_attempts(
    tool_call_log: list[dict],
    requested_doc_ids: list[str],
    filename_map: dict[str, str],
) -> list[SearchAttemptInfo]:
    """Build a list of search attempts including docs that were never searched."""
    attempts: dict[str, SearchAttemptInfo] = {}

    for tc in tool_call_log:
        if tc.get("name") != "search_document":
            continue
        args = tc.get("args", {})
        doc_id = args.get("document_id", "")
        query = args.get("query", "")
        result = tc.get("result") or {}
        results = result.get("results", [])
        has_error = "error" in result

        if has_error:
            s = "error"
        elif results:
            s = "found"
        else:
            s = "no_results"

        attempts[doc_id] = SearchAttemptInfo(
            document_id=doc_id,
            filename=filename_map.get(doc_id, result.get("filename", doc_id)),
            query=query,
            results_found=len(results),
            status=s,
        )

    # Add entries for requested docs that were never searched
    for did in requested_doc_ids:
        if did not in attempts:
            attempts[did] = SearchAttemptInfo(
                document_id=did,
                filename=filename_map.get(did, did),
                status="not_searched",
            )

    return list(attempts.values())


# ---------------------------------------------------------------------------
# Conversation CRUD Routes
# ---------------------------------------------------------------------------

@router.get(
    "/conversations",
    response_model=list[ConversationInfo],
    summary="List conversations",
)
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .where(Conversation.owner_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    rows = (await db.execute(stmt)).scalars().all()
    out = []
    for c in rows:
        # Count messages efficiently
        msg_count_stmt = (
            select(ConversationMessage.id)
            .where(ConversationMessage.conversation_id == c.id)
        )
        msg_count = len((await db.execute(msg_count_stmt)).all())
        out.append(ConversationInfo(
            id=c.id,
            title=c.title,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
            message_count=msg_count,
        ))
    return out


@router.post(
    "/conversations",
    response_model=ConversationInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    body: ConversationCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = Conversation(title=body.title, owner_id=user.id)
    db.add(conv)
    await db.flush()
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        message_count=0,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation with messages",
)
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.owner_id == user.id)
    )
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = []
    for m in conv.messages:
        msgs.append(ConversationMessageInfo(
            id=m.id,
            role=m.role,
            content=m.content,
            tool_name=m.tool_name,
            sources=_sources_from_json(m.sources_json) or None,
            source_type=m.source_type,
            created_at=m.created_at.isoformat(),
        ))
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        messages=msgs,
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationInfo,
    summary="Rename a conversation",
)
async def rename_conversation(
    conversation_id: str,
    body: ConversationRename,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.owner_id == user.id
    )
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.title = body.title
    await db.flush()
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.owner_id == user.id
    )
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.flush()


# ---------------------------------------------------------------------------
# Chat Route (updated with conversation persistence)
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

    If ``conversation_id`` is provided the prior messages are loaded into
    the agent's memory so it has context across turns.  New messages
    (user + tool + assistant) are persisted to the DB after the agent
    responds.
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

    # Resolve filenames for all combined doc IDs
    filename_map: dict[str, str] = {}
    if combined_doc_ids:
        fn_stmt = select(Document.id, Document.filename).where(
            Document.id.in_(combined_doc_ids)
        )
        fn_rows = await db.execute(fn_stmt)
        filename_map = {r[0]: r[1] for r in fn_rows}

    # -- Load conversation history if provided --------------------------------
    conversation_id = body.conversation_id
    memory = ConversationMemory()

    if conversation_id:
        conv_stmt = select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == user.id
        )
        conv = (await db.execute(conv_stmt)).scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Load prior messages into memory
        msgs_stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at)
        )
        prior_msgs = (await db.execute(msgs_stmt)).scalars().all()
        for pm in prior_msgs:
            if pm.role in ("user", "assistant"):
                memory.add_message(pm.role, pm.content or "")
            elif pm.role == "tool":
                memory.add_message("tool", pm.content or "", tool_name=pm.tool_name)

    # -- Run agent ------------------------------------------------------------
    agent = IDPAgent()

    try:
        result = await agent.chat(
            message=body.message,
            document_ids=combined_doc_ids,
            llm=llm,
            db=db,
            conversation=memory,
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
    source_type = _classify_source_type(tool_calls_log, combined_doc_ids)
    search_attempts = _extract_search_attempts(tool_calls_log, combined_doc_ids, filename_map)

    # -- Persist messages to DB -----------------------------------------------
    if conversation_id:
        # Save user message
        db.add(ConversationMessage(
            conversation_id=conversation_id,
            owner_id=user.id,
            role="user",
            content=body.message,
        ))

        # Save tool messages
        for tc in tool_calls_log:
            db.add(ConversationMessage(
                conversation_id=conversation_id,
                owner_id=user.id,
                role="tool",
                content=json.dumps(tc.get("result"), default=str)[:5000] if tc.get("result") else None,
                tool_name=tc.get("name"),
            ))

        # Save assistant message with sources and source_type
        db.add(ConversationMessage(
            conversation_id=conversation_id,
            owner_id=user.id,
            role="assistant",
            content=result["response"],
            sources_json=_sources_to_json(sources),
            source_type=source_type,
        ))

        # Auto-title from first user message
        conv_stmt2 = select(Conversation).where(Conversation.id == conversation_id)
        conv_obj = (await db.execute(conv_stmt2)).scalar_one_or_none()
        if conv_obj and conv_obj.title == "New conversation":
            conv_obj.title = body.message[:100]

        await db.flush()

    return ChatResponse(
        response=result["response"],
        conversation_id=conversation_id,
        tool_calls=tool_calls,
        sources=sources,
        source_type=source_type,
        search_attempts=search_attempts,
    )
