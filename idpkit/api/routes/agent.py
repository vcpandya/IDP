"""IDP Kit Agent API routes — conversational AI agent with tool-calling."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User
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


class ToolCallInfo(BaseModel):
    name: str
    args: dict
    result: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


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
    agent = IDPAgent()

    try:
        result = await agent.chat(
            message=body.message,
            document_ids=body.document_ids,
            llm=llm,
            db=db,
        )
    except Exception as exc:
        logger.error("Agent chat failed for user %s: %s", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing failed: {exc}",
        )

    tool_calls = [
        ToolCallInfo(
            name=tc["name"],
            args=tc["args"],
            result=tc.get("result"),
        )
        for tc in result.get("tool_calls", [])
    ]

    return ChatResponse(
        response=result["response"],
        tool_calls=tool_calls,
    )
