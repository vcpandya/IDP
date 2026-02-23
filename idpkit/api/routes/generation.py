"""IDP Kit Generation API routes — report generation and template management."""

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

router = APIRouter(prefix="/api/generation", tags=["generation"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    document_id: str = Field(..., description="ID of the document to generate a report from")
    template_id: Optional[str] = Field(None, description="Optional template ID to use")
    format: str = Field("markdown", description="Output format: 'docx' or 'markdown'")


class GenerateReportResponse(BaseModel):
    status: str
    format: str
    content: Optional[str] = None
    output_path: Optional[str] = None
    document_id: str


class SaveTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Template name")
    content: str = Field(..., min_length=1, description="Template content")
    format: str = Field("md", description="Template format: 'docx' or 'md'")
    description: Optional[str] = Field(None, description="Template description")


class TemplateResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    format: Optional[str] = None
    content: Optional[str] = None
    created_at: Optional[str] = None


class TemplateListResponse(BaseModel):
    items: list[TemplateResponse]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/report",
    response_model=GenerateReportResponse,
    summary="Generate a report from a document",
)
async def generate_report(
    body: GenerateReportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Generate a report from a document's tree index.

    Supports output in DOCX or Markdown format.  Optionally uses a saved
    template to structure the output.
    """
    # Fetch the document
    result = await db.execute(
        select(Document).where(Document.id == body.document_id, Document.owner_id == user.id)
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

    tree_index = doc.tree_index
    output_format = body.format.lower()

    # Optionally load template
    template_content = None
    if body.template_id:
        from idpkit.generation.templates import get_template
        template_data = await get_template(body.template_id, db)
        if not template_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )
        template_content = template_data.get("content")

    if output_format == "markdown":
        from idpkit.generation.markdown_generator import generate_markdown
        options = {"include_summaries": True, "include_text": False}
        content = generate_markdown(tree_index, options)
        return GenerateReportResponse(
            status="success",
            format="markdown",
            content=content,
            document_id=body.document_id,
        )
    elif output_format == "docx":
        from idpkit.generation.markdown_generator import generate_markdown
        from idpkit.generation.docx_generator import generate_docx

        # First generate markdown content, then convert to docx.
        options = {"include_summaries": True, "include_text": False}
        md_content = generate_markdown(tree_index, options)

        import os
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="idpkit_")
        output_path = os.path.join(output_dir, f"{doc.filename}_report.docx")

        template_path = None
        if template_content:
            # If template is a file path, use it directly.
            template_data_obj = await get_template(body.template_id, db) if body.template_id else {}
            tp = template_data_obj.get("file_path")
            if tp and os.path.isfile(tp):
                template_path = tp

        abs_path = generate_docx(md_content, output_path, template_path=template_path)

        return GenerateReportResponse(
            status="success",
            format="docx",
            output_path=abs_path,
            document_id=body.document_id,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {output_format!r}. Use 'docx' or 'markdown'.",
        )


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="List available templates",
)
async def list_templates_route(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all templates owned by the current user."""
    from idpkit.generation.templates import list_templates
    templates = await list_templates(db, owner_id=user.id)
    return TemplateListResponse(items=templates)


@router.post(
    "/templates",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a new template",
)
async def save_template_route(
    body: SaveTemplateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a new document template."""
    from idpkit.generation.templates import save_template
    result = await save_template(
        name=body.name,
        content=body.content,
        format=body.format,
        db=db,
        owner_id=user.id,
        description=body.description,
    )
    return result
