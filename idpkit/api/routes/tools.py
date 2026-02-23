"""IDP Kit Tools API routes — execute Smart Tools on documents."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User
from idpkit.api.deps import get_current_user, get_llm
from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ToolExecuteRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to process")
    options: dict = Field(default_factory=dict, description="Tool-specific options")
    model: Optional[str] = Field(None, description="LLM model override")


class ToolInfo(BaseModel):
    name: str
    display_name: str
    description: str
    options_schema: dict


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
    total: int


class ToolExecuteResponse(BaseModel):
    tool_name: str
    status: str
    data: Optional[dict] = None
    output_file: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=ToolListResponse,
    summary="List all available Smart Tools",
)
async def list_tools(
    user: User = Depends(get_current_user),
):
    """Return metadata for all registered Smart Tools."""
    from idpkit.tools import TOOL_REGISTRY

    tools = []
    for name, tool_instance in TOOL_REGISTRY.items():
        tools.append(
            ToolInfo(
                name=tool_instance.name,
                display_name=tool_instance.display_name,
                description=tool_instance.description,
                options_schema=tool_instance.options_schema,
            )
        )

    return ToolListResponse(tools=tools, total=len(tools))


@router.post(
    "/{tool_name}",
    response_model=ToolExecuteResponse,
    summary="Execute a Smart Tool on a document",
)
async def execute_tool(
    tool_name: str,
    body: ToolExecuteRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Execute a specific Smart Tool on a document.

    The tool is identified by its ``tool_name`` in the URL path.
    Tool-specific options are passed in the request body.
    """
    from idpkit.tools import TOOL_REGISTRY

    tool_instance = TOOL_REGISTRY.get(tool_name)
    if not tool_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found. Use GET /api/tools/ to list available tools.",
        )

    # Apply model override if provided
    tool_llm = llm
    if body.model:
        tool_llm = LLMClient(
            default_model=body.model,
            api_key=llm.api_key,
            api_base=llm.api_base,
        )

    try:
        result = await tool_instance.execute(
            document_id=body.document_id,
            options=body.options,
            llm=tool_llm,
            db=db,
        )
    except Exception as exc:
        logger.error(
            "Tool '%s' failed for document %s: %s",
            tool_name,
            body.document_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {exc}",
        )

    return ToolExecuteResponse(
        tool_name=result.tool_name,
        status=result.status,
        data=result.data,
        output_file=result.output_file,
        error=result.error,
    )
