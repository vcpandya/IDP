"""IDP Kit Agent API routes — conversational AI agent with tool-calling."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.db.session import get_db
from idpkit.db.models import (
    User, Document, document_tags, Conversation, ConversationMessage,
    Tag, conversation_tags,
)
from idpkit.api.deps import get_current_user, get_llm, get_llm_for_user
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
    document_id: str = ""
    filename: str = ""
    node_id: Optional[str] = None
    title: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    summary: Optional[str] = None
    text_preview: Optional[str] = None
    url: Optional[str] = None
    source_kind: str = "document"


class SearchAttemptInfo(BaseModel):
    document_id: str
    filename: str
    query: str = ""
    results_found: int = 0
    status: str = "not_searched"  # found, no_results, error, not_searched


class ChatComputation(BaseModel):
    """A Python execution captured during the agent loop, surfaced to the UI
    so we can highlight numbers IDA computed and let the user inspect the
    code, stdout, and any chart it produced.
    """
    cid: str  # "py1", "py2", … (1-indexed in tool-call order)
    code: str
    stdout: str = ""
    stderr: str = ""
    results: list[str] = Field(default_factory=list)
    charts: list[dict] = Field(default_factory=list)  # {type, data(base64)}
    success: bool = True
    error: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    sources: list[ChatSourceInfo] = Field(default_factory=list)
    source_type: str = "general_knowledge"  # documents, general_knowledge, mixed
    search_attempts: list[SearchAttemptInfo] = Field(default_factory=list)
    computations: list[ChatComputation] = Field(default_factory=list)


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
    computations: Optional[list[ChatComputation]] = None
    created_at: str


class ConversationTagInfo(BaseModel):
    id: str
    name: str
    color: Optional[str] = None


class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    tags: list[ConversationTagInfo] = Field(default_factory=list)


class ConversationTagsUpdate(BaseModel):
    tag_ids: list[str] = Field(default_factory=list)


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessageInfo] = Field(default_factory=list)
    tags: list[ConversationTagInfo] = Field(default_factory=list)


def _conv_tags(conv: Conversation) -> list[ConversationTagInfo]:
    return [
        ConversationTagInfo(id=t.id, name=t.name, color=t.color)
        for t in (conv.tags or [])
    ]


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
             start_page: int | None, end_page: int | None, summary: str | None = None,
             text_preview: str | None = None):
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
            text_preview=(text_preview or "")[:3000] or None,
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
                     node.get("end_page"), node.get("summary"),
                     node.get("text_preview"))

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

        elif name == "web_search":
            for item in result.get("results", []):
                item_url = item.get("url", "")
                if not item_url:
                    continue
                key = ("web", item_url)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(ChatSourceInfo(
                    title=item.get("title", ""),
                    summary=(item.get("description", "") or "")[:200] or None,
                    url=item_url,
                    source_kind="web",
                ))

        elif name == "fetch_url":
            item_url = result.get("url", "")
            if item_url:
                key = ("web", item_url)
                if key not in seen:
                    seen.add(key)
                    sources.append(ChatSourceInfo(
                        title=result.get("title", "") or item_url,
                        url=item_url,
                        source_kind="web",
                    ))

    return sources


def _extract_computations(tool_call_log: list[dict]) -> list[ChatComputation]:
    """Pull every execute_python call from the loop in order, assigning
    sequential `pyN` IDs that line up with the `[[py]]` markers IDA emits.
    """
    out: list[ChatComputation] = []
    counter = 0
    for tc in tool_call_log:
        if tc.get("name") != "execute_python":
            continue
        counter += 1
        args = tc.get("args") or {}
        result = tc.get("result") or {}
        # Charts are base64 PNG/SVG and can be very large; cap to keep
        # responses and DB rows reasonable. Two charts per call is plenty
        # for an inspection modal.
        charts_in = result.get("charts") or []
        charts_out: list[dict] = []
        for ch in charts_in[:2]:
            data = ch.get("data") or ""
            if isinstance(data, str) and len(data) > 350_000:
                # Skip oversized charts entirely rather than silently
                # truncating base64 (which would corrupt the image).
                continue
            charts_out.append({"type": ch.get("type"), "data": data})
        out.append(ChatComputation(
            cid=f"py{counter}",
            code=(args.get("code") or "")[:8000],
            stdout=(result.get("stdout") or "")[:4000],
            stderr=(result.get("stderr") or "")[:2000],
            results=[str(r)[:1000] for r in (result.get("results") or [])][:5],
            charts=charts_out,
            success=bool(result.get("success", "error" not in result)),
            error=result.get("error") if isinstance(result.get("error"), dict) else None,
        ))
    return out


def _computations_to_json(items: list[ChatComputation]) -> list[dict] | None:
    """Serialize computations for the DB. Strip chart payloads (we keep a
    flag indicating one existed) so we don't bloat the JSON column with
    base64 images for every saved message."""
    if not items:
        return None
    out: list[dict] = []
    for c in items:
        d = c.model_dump(exclude_none=True)
        # Replace bulky chart bytes with a lightweight stub on persistence.
        if d.get("charts"):
            d["charts"] = [{"type": ch.get("type"), "data": ""} for ch in d["charts"]]
        out.append(d)
    return out


def _computations_from_json(data) -> list[ChatComputation]:
    if not data:
        return []
    return [ChatComputation(**c) for c in data]


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
    search_doc_ids: set[str] = set()
    found_doc_ids: set[str] = set()
    used_web = False

    for tc in tool_call_log:
        name = tc.get("name", "")
        if name == "search_document":
            doc_id = tc.get("args", {}).get("document_id", "")
            if doc_id:
                search_doc_ids.add(doc_id)
            result = tc.get("result") or {}
            results = result.get("results", [])
            if results and doc_id:
                found_doc_ids.add(doc_id)
        elif name in ("web_search", "fetch_url"):
            result = tc.get("result") or {}
            if name == "web_search" and result.get("results"):
                used_web = True
            elif name == "fetch_url" and result.get("url"):
                used_web = True

    has_docs = bool(found_doc_ids)
    if has_docs and used_web:
        return "mixed"
    if has_docs:
        return "documents"
    if used_web:
        return "web"
    if not requested_doc_ids:
        return "general_knowledge"
    if search_doc_ids and not found_doc_ids:
        return "general_knowledge"
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
    # Lean query: just metadata + tags. We deliberately do NOT compute
    # message_count here — it required a JOIN+GROUP_BY across the entire
    # message table per request and was the main reason this endpoint was
    # slow on chat-heavy accounts. The sidebar only renders the title +
    # tags, so message_count is left at its default of 0.
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.tags))
        .where(Conversation.owner_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        ConversationInfo(
            id=c.id,
            title=c.title,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
            tags=_conv_tags(c),
        )
        for c in rows
    ]


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
    await db.commit()
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        message_count=0,
        tags=[],
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
        .options(
            selectinload(Conversation.messages),
            selectinload(Conversation.tags),
        )
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
            computations=_computations_from_json(m.computations_json) or None,
            created_at=m.created_at.isoformat(),
        ))
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        messages=msgs,
        tags=_conv_tags(conv),
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
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.tags))
        .where(Conversation.id == conversation_id, Conversation.owner_id == user.id)
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
        tags=_conv_tags(conv),
    )


@router.put(
    "/conversations/{conversation_id}/tags",
    response_model=ConversationInfo,
    summary="Set tags on a conversation",
)
async def set_conversation_tags(
    conversation_id: str,
    body: ConversationTagsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.tags))
        .where(Conversation.id == conversation_id, Conversation.owner_id == user.id)
    )
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.tag_ids:
        # De-dupe so an accidental repeated id doesn't make the count check
        # fail when every id is otherwise valid.
        requested_ids = list(dict.fromkeys(body.tag_ids))
        tag_rows = (await db.execute(
            select(Tag).where(
                Tag.id.in_(requested_ids),
                Tag.owner_id == user.id,
            )
        )).scalars().all()
        if len(tag_rows) != len(requested_ids):
            found = {t.id for t in tag_rows}
            missing = [tid for tid in requested_ids if tid not in found]
            raise HTTPException(
                status_code=400,
                detail=f"Unknown or inaccessible tag id(s): {missing}",
            )
        conv.tags = list(tag_rows)
    else:
        conv.tags = []
    await db.flush()
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        tags=_conv_tags(conv),
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

from idpkit.api.deps import limiter, get_rate_limit

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the IDP Agent",
)
@limiter.limit(lambda: get_rate_limit("agent_chat"))
async def agent_chat(
    request: Request,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a message to the IDP Agent and receive a response.

    If ``conversation_id`` is provided the prior messages are loaded into
    the agent's memory so it has context across turns.  New messages
    (user + tool + assistant) are persisted to the DB after the agent
    responds.
    """
    llm = get_llm_for_user(user)
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

    logger.info(
        "Agent chat: user=%s convo=%s docs=%d tags=%d",
        user.id, conversation_id or "-",
        len(combined_doc_ids), len(body.tag_ids),
    )

    # -- Run agent ------------------------------------------------------------
    agent = IDPAgent()

    try:
        result = await agent.chat(
            message=body.message,
            document_ids=combined_doc_ids,
            llm=llm,
            db=db,
            conversation=memory,
            user_id=user.id,
        )
    except Exception as exc:
        logger.error("Agent chat failed for user %s: %s", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent processing failed",
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
    computations = _extract_computations(tool_calls_log)

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

        # Save assistant message with sources, source_type, and computations
        db.add(ConversationMessage(
            conversation_id=conversation_id,
            owner_id=user.id,
            role="assistant",
            content=result["response"],
            sources_json=_sources_to_json(sources),
            source_type=source_type,
            computations_json=_computations_to_json(computations),
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
        computations=computations,
    )


# ---------------------------------------------------------------------------
# Streaming chat — SSE
# ---------------------------------------------------------------------------

@router.post(
    "/chat/stream",
    summary="Chat with the IDP Agent (SSE stream)",
)
@limiter.limit(lambda: get_rate_limit("agent_chat"))
async def agent_chat_stream(
    request: Request,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Server-Sent Events variant of :func:`agent_chat`.

    Emits the same lifecycle events the agent yields (`thinking`,
    `tool_start`, `tool_end`, `text_delta`) plus a final `done` event
    whose payload mirrors :class:`ChatResponse` so the UI can finalize
    sources, computations, and persistence with the same logic.

    The classic ``/api/agent/chat`` endpoint is left untouched for any
    non-streaming caller.
    """
    llm = get_llm_for_user(user)

    # Resolve tag_ids → document_ids (same as /chat)
    combined_doc_ids = list(body.document_ids)
    if body.tag_ids:
        stmt = select(document_tags.c.document_id).where(
            document_tags.c.tag_id.in_(body.tag_ids)
        )
        rows = await db.execute(stmt)
        for did in [r[0] for r in rows]:
            if did not in combined_doc_ids:
                combined_doc_ids.append(did)

    filename_map: dict[str, str] = {}
    if combined_doc_ids:
        fn_rows = await db.execute(
            select(Document.id, Document.filename).where(
                Document.id.in_(combined_doc_ids)
            )
        )
        filename_map = {r[0]: r[1] for r in fn_rows}

    # Conversation history
    conversation_id = body.conversation_id
    memory = ConversationMemory()
    if conversation_id:
        conv = (await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.owner_id == user.id,
            )
        )).scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        prior_msgs = (await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at)
        )).scalars().all()
        for pm in prior_msgs:
            if pm.role in ("user", "assistant"):
                memory.add_message(pm.role, pm.content or "")
            elif pm.role == "tool":
                memory.add_message("tool", pm.content or "", tool_name=pm.tool_name)

    logger.info(
        "Agent chat stream: user=%s convo=%s docs=%d tags=%d",
        user.id, conversation_id or "-",
        len(combined_doc_ids), len(body.tag_ids),
    )

    agent = IDPAgent()

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, default=str)}\n\n"

    async def event_source():
        # Emit a tiny keepalive comment up-front so the browser flushes
        # response headers immediately and the user sees activity.
        yield ": stream-open\n\n"

        final_event: dict | None = None
        agen = agent.chat_stream(
            message=body.message,
            document_ids=combined_doc_ids,
            llm=llm,
            db=db,
            conversation=memory,
            user_id=user.id,
        )
        try:
            async for ev in agen:
                # Bail out early if the client has gone away. This stops us
                # from chewing through expensive long-running tools (notably
                # deep_research, which polls up to 5 minutes) for nobody.
                if await request.is_disconnected():
                    logger.info(
                        "Agent stream client disconnected (user=%s convo=%s) — aborting.",
                        user.id, conversation_id or "-",
                    )
                    try:
                        await agen.aclose()
                    except Exception:
                        pass
                    return

                if ev.get("type") == "done":
                    final_event = ev
                    # Don't forward the raw `done` yet — we want to enrich it
                    # with sources / computations / persistence first.
                    continue
                yield _sse(ev)
        except Exception as exc:
            logger.error("Agent stream failed for user %s: %s", user.id, exc)
            yield _sse({"type": "error", "message": "Agent processing failed"})
            yield _sse({"type": "done", "response": "", "tool_calls": []})
            return

        # Build the enriched done payload (same shape as ChatResponse)
        tool_calls_log = (final_event or {}).get("tool_calls", [])
        response_text = (final_event or {}).get("response", "")

        sources = _extract_sources(tool_calls_log)
        source_type = _classify_source_type(tool_calls_log, combined_doc_ids)
        search_attempts = _extract_search_attempts(
            tool_calls_log, combined_doc_ids, filename_map,
        )
        computations = _extract_computations(tool_calls_log)

        # Persist to DB exactly like /chat does. Persistence parity matters:
        # if the save fails we MUST tell the client (otherwise the user sees
        # a successful answer that quietly disappears on reload, and the
        # next turn loses conversation state). Mirror /chat's contract by
        # surfacing an explicit error event before the done event.
        persistence_error: str | None = None
        if conversation_id:
            try:
                db.add(ConversationMessage(
                    conversation_id=conversation_id,
                    owner_id=user.id,
                    role="user",
                    content=body.message,
                ))
                for tc in tool_calls_log:
                    db.add(ConversationMessage(
                        conversation_id=conversation_id,
                        owner_id=user.id,
                        role="tool",
                        content=json.dumps(tc.get("result"), default=str)[:5000]
                            if tc.get("result") else None,
                        tool_name=tc.get("name"),
                    ))
                db.add(ConversationMessage(
                    conversation_id=conversation_id,
                    owner_id=user.id,
                    role="assistant",
                    content=response_text,
                    sources_json=_sources_to_json(sources),
                    source_type=source_type,
                    computations_json=_computations_to_json(computations),
                ))
                conv_obj = (await db.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )).scalar_one_or_none()
                if conv_obj and conv_obj.title == "New conversation":
                    conv_obj.title = body.message[:100]
                await db.flush()
            except Exception as exc:
                logger.error("Stream persistence failed: %s", exc)
                persistence_error = (
                    "Your reply was generated but could not be saved to "
                    "history. Please refresh and try again."
                )
                yield _sse({"type": "error", "message": persistence_error})

        yield _sse({
            "type": "done",
            "response": response_text,
            "persistence_error": persistence_error,
            "conversation_id": conversation_id,
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"], "result": tc.get("result")}
                for tc in tool_calls_log
            ],
            "sources": [s.model_dump() if hasattr(s, "model_dump") else s for s in sources],
            "source_type": source_type,
            "search_attempts": [
                a.model_dump() if hasattr(a, "model_dump") else a for a in search_attempts
            ],
            "computations": [
                c.model_dump() if hasattr(c, "model_dump") else c for c in computations
            ],
        })

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
