"""IDP Kit Batch Processing API — templates, batch jobs, upload-and-process."""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import (
    BatchItem,
    BatchJob,
    Document,
    ProcessingTemplate,
    User,
)
from idpkit.api.deps import get_current_user, get_llm, get_storage
from idpkit.core.llm import LLMClient
from idpkit.core.storage import StorageBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/batch", tags=["batch"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TemplateCreateRequest(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tool_name: str = "custom"
    tool_options: Optional[dict] = None
    reference_content: Optional[str] = None
    output_format: str = "json"
    output_template: Optional[dict] = None
    model: Optional[str] = None
    is_public: int = 0


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tool_name: Optional[str] = None
    tool_options: Optional[dict] = None
    reference_content: Optional[str] = None
    output_format: Optional[str] = None
    output_template: Optional[dict] = None
    model: Optional[str] = None
    is_public: Optional[int] = None


class TemplateResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tool_name: str
    tool_options: Optional[dict] = None
    reference_content: Optional[str] = None
    output_format: str = "json"
    output_template: Optional[dict] = None
    model: Optional[str] = None
    owner_id: str
    is_public: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BatchCreateRequest(BaseModel):
    template_id: Optional[str] = None
    tool_name: Optional[str] = None
    document_ids: list[str]
    prompt: Optional[str] = None
    options: Optional[dict] = None
    model: Optional[str] = None
    concurrency: int = Field(3, ge=1, le=10)
    reference_doc_ids: Optional[list[str]] = None


class BatchItemResponse(BaseModel):
    id: str
    document_id: str
    position: int = 0
    status: str = "pending"
    result: Optional[dict] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    filename: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchJobResponse(BaseModel):
    id: str
    owner_id: str
    template_id: Optional[str] = None
    tool_name: str
    status: str = "pending"
    progress: int = 0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    prompt: Optional[str] = None
    options: Optional[dict] = None
    model: Optional[str] = None
    concurrency: int = 3
    result_summary: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    items: Optional[list[BatchItemResponse]] = None

    class Config:
        from_attributes = True


class BatchJobListResponse(BaseModel):
    jobs: list[BatchJobResponse]
    total: int


class MessageResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Template CRUD endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/templates",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a processing template",
)
async def create_template(
    body: TemplateCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Validate tool_name
    if body.tool_name != "custom":
        from idpkit.tools import TOOL_REGISTRY
        if body.tool_name not in TOOL_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown tool '{body.tool_name}'. Use 'custom' or a registered tool name.",
            )

    tmpl = ProcessingTemplate(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        tool_name=body.tool_name,
        tool_options=body.tool_options,
        reference_content=body.reference_content,
        output_format=body.output_format,
        output_template=body.output_template,
        model=body.model,
        owner_id=user.id,
        is_public=body.is_public,
    )
    db.add(tmpl)
    await db.flush()
    await db.refresh(tmpl)
    return tmpl


@router.get(
    "/templates",
    response_model=list[TemplateResponse],
    summary="List processing templates",
)
async def list_templates(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProcessingTemplate)
        .where(
            (ProcessingTemplate.owner_id == user.id) | (ProcessingTemplate.is_public == 1)
        )
        .order_by(ProcessingTemplate.created_at.desc())
        .limit(200)
    )
    return result.scalars().all()


@router.get(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Get template details",
)
async def get_template(
    template_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProcessingTemplate).where(
            ProcessingTemplate.id == template_id,
            (ProcessingTemplate.owner_id == user.id) | (ProcessingTemplate.is_public == 1),
        )
    )
    tmpl = result.scalar_one_or_none()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return tmpl


@router.put(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Update a template",
)
async def update_template(
    template_id: str,
    body: TemplateUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProcessingTemplate).where(
            ProcessingTemplate.id == template_id,
            ProcessingTemplate.owner_id == user.id,
        )
    )
    tmpl = result.scalar_one_or_none()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")

    update_data = body.model_dump(exclude_unset=True)
    # Validate tool_name if being changed
    if "tool_name" in update_data and update_data["tool_name"] != "custom":
        from idpkit.tools import TOOL_REGISTRY
        if update_data["tool_name"] not in TOOL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool '{update_data['tool_name']}'",
            )

    for field, value in update_data.items():
        setattr(tmpl, field, value)

    await db.flush()
    await db.refresh(tmpl)
    return tmpl


@router.delete(
    "/templates/{template_id}",
    response_model=MessageResponse,
    summary="Delete a template",
)
async def delete_template(
    template_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProcessingTemplate).where(
            ProcessingTemplate.id == template_id,
            ProcessingTemplate.owner_id == user.id,
        )
    )
    tmpl = result.scalar_one_or_none()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")
    await db.delete(tmpl)
    await db.flush()
    return MessageResponse(detail=f"Template '{tmpl.name}' deleted")


@router.post(
    "/templates/analyze",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="AI-generate a template from a sample document",
)
async def analyze_template(
    sample_file: UploadFile = File(...),
    template_name: str = Form(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Upload a sample document and let AI analyze its structure to create a template."""
    if not sample_file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Extract text from the sample
    content = await sample_file.read()
    ext = os.path.splitext(sample_file.filename)[1].lower()

    text = ""
    try:
        from idpkit.indexing import get_indexer
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            indexer = get_indexer(ext)
            result = await indexer.build_index(tmp_path)
            # Extract text from structure
            from idpkit.batch.runner import _extract_text_from_tree
            text = _extract_text_from_tree(result.get("structure"))
        finally:
            os.unlink(tmp_path)
    except Exception as exc:
        logger.warning("Failed to index sample for analysis: %s", exc)
        # Try raw text for common formats
        try:
            text = content.decode("utf-8", errors="ignore")[:10000]
        except Exception:
            raise HTTPException(status_code=400, detail="Could not extract text from sample document")

    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the sample")

    # Ask LLM to analyze structure and create template
    analyze_prompt = f"""Analyze this document and create a processing template configuration.

Document text (first 8000 chars):
{text[:8000]}

Return a JSON object with:
{{
    "description": "What this template does (1-2 sentences)",
    "system_prompt": "A detailed system prompt for processing similar documents",
    "output_template": {{
        "sections": [
            {{
                "name": "section name",
                "heading_level": 1,
                "description": "what goes in this section"
            }}
        ],
        "formatting_rules": {{
            "header_style": "bold",
            "bullet_style": "dash",
            "include_page_numbers": false
        }},
        "structure_elements": ["title", "sections identified in the document"]
    }}
}}

Return ONLY valid JSON, no other text."""

    try:
        response = await llm.acomplete(analyze_prompt)
        resp_text = response.content.strip()
        if resp_text.startswith("```"):
            resp_text = resp_text.split("\n", 1)[1] if "\n" in resp_text else resp_text[3:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            resp_text = resp_text.strip()

        analysis = json.loads(resp_text)
    except (json.JSONDecodeError, Exception) as exc:
        logger.exception("Failed to analyze document")
        raise HTTPException(status_code=500, detail="Failed to analyze document")

    tmpl = ProcessingTemplate(
        name=template_name,
        description=analysis.get("description", "AI-generated template"),
        system_prompt=analysis.get("system_prompt", ""),
        tool_name="custom",
        output_format="json",
        output_template=analysis.get("output_template"),
        owner_id=user.id,
    )
    db.add(tmpl)
    await db.flush()
    await db.refresh(tmpl)
    return tmpl


# ---------------------------------------------------------------------------
# Options conversion (plain text → structured JSON via LLM)
# ---------------------------------------------------------------------------

class ConvertOptionsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    tool_name: Optional[str] = "custom"


class ConvertOptionsResponse(BaseModel):
    options: dict
    schema_used: dict
    raw_text: str


@router.post(
    "/convert-options",
    response_model=ConvertOptionsResponse,
    summary="Convert plain-text options to structured JSON via AI (two-pass)",
)
async def convert_options(
    body: ConvertOptionsRequest,
    user: User = Depends(get_current_user),
    llm: LLMClient = Depends(get_llm),
):
    """Two-pass conversion of free-text processing preferences into structured JSON.

    Pass 1: Analyze the user's text and generate a JSON Schema that captures
    all the preferences they expressed — no predefined fields.

    Pass 2: Use that generated schema with structured output to produce the
    final options JSON object.
    """
    import json as _json

    # ------------------------------------------------------------------
    # Pass 1: Generate a JSON Schema from the user's text
    # ------------------------------------------------------------------
    pass1_system = (
        "You are a schema architect. The user has described processing preferences "
        "in plain text. Analyze what they want and generate a JSON Schema that "
        "captures every preference they expressed as typed fields.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON Schema (no markdown, no explanation)\n"
        "- Create field names that are descriptive snake_case keys\n"
        "- Use appropriate types: string, number, integer, boolean, array, object\n"
        "- Add 'description' to each property explaining what it captures\n"
        "- Use 'enum' where the preference implies a fixed set of choices\n"
        "- Include 'type': 'object' at the root with 'properties'\n"
        "- Only create fields for things the user actually mentioned"
    )

    generated_schema = None
    try:
        pass1_response = await llm.acomplete(
            f"Generate a JSON Schema for these processing preferences:\n\n{body.text}",
            chat_history=[{"role": "system", "content": pass1_system}],
        )
        raw = pass1_response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        schema = _json.loads(raw)
        if isinstance(schema, dict) and "properties" in schema:
            generated_schema = schema
            logger.info(
                "Options Pass 1: Generated schema with %d properties",
                len(schema.get("properties", {})),
            )
    except Exception as exc:
        logger.warning("Options Pass 1 (schema generation) failed: %s", exc)

    if not generated_schema:
        return ConvertOptionsResponse(
            options={"instructions": body.text},
            schema_used={},
            raw_text=body.text,
        )

    # ------------------------------------------------------------------
    # Pass 2: Use the generated schema to produce structured options
    # ------------------------------------------------------------------
    schema_str = _json.dumps(generated_schema, indent=2)
    pass2_system = (
        "You convert plain-text processing preferences into a structured JSON object "
        "that conforms to the provided JSON Schema. Map each preference the user "
        "expressed to the appropriate field.\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n\n"
        f"JSON Schema:\n{schema_str}"
    )

    try:
        pass2_response = await llm.acomplete(
            body.text,
            chat_history=[{"role": "system", "content": pass2_system}],
            response_format={"type": "json_object"},
        )
        options = _json.loads(pass2_response.content)
        if not isinstance(options, dict):
            options = {"instructions": body.text}
    except Exception as exc:
        logger.warning("Options Pass 2 (structured extraction) failed: %s", exc)
        options = {"instructions": body.text}

    return ConvertOptionsResponse(
        options=options,
        schema_used=generated_schema,
        raw_text=body.text,
    )


# ---------------------------------------------------------------------------
# Batch job endpoints
# ---------------------------------------------------------------------------

from idpkit.api.deps import limiter, get_rate_limit

@router.post(
    "/",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and start a batch job",
)
@limiter.limit(lambda: get_rate_limit("batch_create"))
async def create_batch(
    request: Request,
    body: BatchCreateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a batch job from existing documents."""
    if not body.document_ids:
        raise HTTPException(status_code=400, detail="At least one document_id is required")

    # Determine tool_name
    tool_name = body.tool_name
    if body.template_id:
        result = await db.execute(
            select(ProcessingTemplate).where(ProcessingTemplate.id == body.template_id)
        )
        tmpl = result.scalar_one_or_none()
        if not tmpl:
            raise HTTPException(status_code=404, detail="Template not found")
        if not tool_name:
            tool_name = tmpl.tool_name

    if not tool_name:
        raise HTTPException(status_code=400, detail="tool_name is required (or use a template)")

    # Validate tool exists
    if tool_name != "custom":
        from idpkit.tools import TOOL_REGISTRY
        if tool_name not in TOOL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown tool '{tool_name}'")

    # Validate documents belong to user
    result = await db.execute(
        select(Document).where(
            Document.id.in_(body.document_ids),
            Document.owner_id == user.id,
        )
    )
    found_docs = {doc.id: doc for doc in result.scalars().all()}
    missing = set(body.document_ids) - set(found_docs.keys())
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {list(missing)}")

    # Create batch job
    batch = BatchJob(
        owner_id=user.id,
        template_id=body.template_id,
        tool_name=tool_name,
        total_items=len(body.document_ids),
        prompt=body.prompt,
        options=body.options,
        model=body.model,
        concurrency=body.concurrency,
        reference_doc_ids=body.reference_doc_ids,
    )
    db.add(batch)
    await db.flush()

    # Create batch items
    for position, doc_id in enumerate(body.document_ids):
        item = BatchItem(
            batch_job_id=batch.id,
            document_id=doc_id,
            position=position,
        )
        db.add(item)
    await db.flush()
    await db.refresh(batch)

    # Launch background processing
    from idpkit.batch.runner import run_batch_job
    background_tasks.add_task(run_batch_job, batch.id)

    return BatchJobResponse(
        id=batch.id,
        owner_id=batch.owner_id,
        template_id=batch.template_id,
        tool_name=batch.tool_name,
        status=batch.status,
        progress=batch.progress,
        total_items=batch.total_items,
        completed_items=batch.completed_items,
        failed_items=batch.failed_items,
        prompt=batch.prompt,
        options=batch.options,
        model=batch.model,
        concurrency=batch.concurrency,
        created_at=batch.created_at,
    )


@router.post(
    "/upload-and-process",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload files and process as a batch",
)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    tool_name: str = Form(...),
    template_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    options: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    auto_index: bool = Form(True),
    concurrency: int = Form(3),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Upload multiple files and immediately process them as a batch."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    # Validate tool
    if tool_name != "custom":
        from idpkit.tools import TOOL_REGISTRY
        if tool_name not in TOOL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown tool '{tool_name}'")

    # Parse options JSON string
    parsed_options = {}
    if options:
        try:
            parsed_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in options field")

    if auto_index:
        parsed_options["_auto_index"] = True

    # Upload each file and create Document records
    from idpkit.api.routes.documents import _detect_format, _storage_key
    doc_ids = []
    for file in files:
        if not file.filename:
            continue
        fmt, ext = _detect_format(file.filename)
        doc = Document(
            filename=file.filename,
            format=fmt,
            owner_id=user.id,
            status="uploaded",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)

        content = await file.read()
        key = _storage_key(user.id, doc.id, ext)
        storage.save(key, content)
        doc.file_path = key
        doc.file_size = len(content)
        await db.flush()
        doc_ids.append(doc.id)

    if not doc_ids:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    # Create batch job
    batch = BatchJob(
        owner_id=user.id,
        template_id=template_id,
        tool_name=tool_name,
        total_items=len(doc_ids),
        prompt=prompt,
        options=parsed_options,
        model=model,
        concurrency=min(max(concurrency, 1), 10),
    )
    db.add(batch)
    await db.flush()

    for position, doc_id in enumerate(doc_ids):
        item = BatchItem(
            batch_job_id=batch.id,
            document_id=doc_id,
            position=position,
        )
        db.add(item)
    await db.flush()
    await db.refresh(batch)

    # Launch background
    from idpkit.batch.runner import run_batch_job
    background_tasks.add_task(run_batch_job, batch.id)

    return BatchJobResponse(
        id=batch.id,
        owner_id=batch.owner_id,
        template_id=batch.template_id,
        tool_name=batch.tool_name,
        status=batch.status,
        progress=batch.progress,
        total_items=batch.total_items,
        completed_items=batch.completed_items,
        failed_items=batch.failed_items,
        prompt=batch.prompt,
        options=batch.options,
        model=batch.model,
        concurrency=batch.concurrency,
        created_at=batch.created_at,
    )


@router.get(
    "/",
    response_model=BatchJobListResponse,
    summary="List batch jobs",
)
async def list_batches(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base = select(BatchJob).where(BatchJob.owner_id == user.id)
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(BatchJob.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    jobs = result.scalars().all()

    return BatchJobListResponse(
        jobs=[
            BatchJobResponse(
                id=j.id,
                owner_id=j.owner_id,
                template_id=j.template_id,
                tool_name=j.tool_name,
                status=j.status,
                progress=j.progress,
                total_items=j.total_items,
                completed_items=j.completed_items,
                failed_items=j.failed_items,
                prompt=j.prompt,
                options=j.options,
                model=j.model,
                concurrency=j.concurrency,
                result_summary=j.result_summary,
                error=j.error,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
            )
            for j in jobs
        ],
        total=total,
    )


@router.get(
    "/{batch_id}",
    response_model=BatchJobResponse,
    summary="Get batch job details",
)
async def get_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    # Load items with document filenames
    items_result = await db.execute(
        select(BatchItem)
        .where(BatchItem.batch_job_id == batch_id)
        .order_by(BatchItem.position)
    )
    items = items_result.scalars().all()

    # Fetch filenames for all document_ids
    doc_ids = [item.document_id for item in items]
    if doc_ids:
        docs_result = await db.execute(
            select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
        )
        doc_names = {row[0]: row[1] for row in docs_result.all()}
    else:
        doc_names = {}

    return BatchJobResponse(
        id=batch.id,
        owner_id=batch.owner_id,
        template_id=batch.template_id,
        tool_name=batch.tool_name,
        status=batch.status,
        progress=batch.progress,
        total_items=batch.total_items,
        completed_items=batch.completed_items,
        failed_items=batch.failed_items,
        prompt=batch.prompt,
        options=batch.options,
        model=batch.model,
        concurrency=batch.concurrency,
        result_summary=batch.result_summary,
        error=batch.error,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
        items=[
            BatchItemResponse(
                id=item.id,
                document_id=item.document_id,
                position=item.position,
                status=item.status,
                result=item.result,
                output_file=item.output_file,
                error=item.error,
                filename=doc_names.get(item.document_id),
                created_at=item.created_at,
                started_at=item.started_at,
                completed_at=item.completed_at,
            )
            for item in items
        ],
    )


@router.get(
    "/{batch_id}/results",
    summary="Get all results from a batch",
)
async def get_batch_results(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    items_result = await db.execute(
        select(BatchItem)
        .where(BatchItem.batch_job_id == batch_id)
        .order_by(BatchItem.position)
    )
    items = items_result.scalars().all()

    doc_ids = [item.document_id for item in items]
    if doc_ids:
        docs_result = await db.execute(
            select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
        )
        doc_names = {row[0]: row[1] for row in docs_result.all()}
    else:
        doc_names = {}

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "results": [
            {
                "document_id": item.document_id,
                "filename": doc_names.get(item.document_id),
                "status": item.status,
                "result": item.result,
                "output_file": item.output_file,
                "error": item.error,
            }
            for item in items
        ],
    }


@router.get(
    "/{batch_id}/items/{item_id}/download",
    summary="Download a single batch item result",
)
async def download_batch_item(
    batch_id: str,
    item_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from fastapi.responses import Response

    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    item_result = await db.execute(
        select(BatchItem).where(BatchItem.id == item_id, BatchItem.batch_job_id == batch_id)
    )
    item = item_result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Batch item not found")

    if item.status != "completed":
        raise HTTPException(status_code=400, detail="No result available for this item")

    doc_result = await db.execute(
        select(Document.filename).where(Document.id == item.document_id)
    )
    filename = doc_result.scalar_one_or_none() or item.document_id
    safe_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    content = None
    is_docx = False
    if item.output_file and item.output_file.endswith(".docx"):
        try:
            storage = get_storage()
            content = storage.load(item.output_file)
            is_docx = True
        except Exception:
            pass
    if content is None and item.output_file:
        try:
            storage = get_storage()
            content = storage.load(item.output_file)
        except Exception:
            pass
    if content is None and item.result is not None:
        try:
            from idpkit.batch.formatter import format_result_to_docx
            content = format_result_to_docx(batch.tool_name, item.result, filename=filename)
            is_docx = True
        except Exception:
            content = json.dumps(item.result, indent=2, default=str).encode("utf-8")
    if content is None:
        raise HTTPException(status_code=400, detail="No result available for this item")

    if is_docx:
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}_result.docx"'
            },
        )
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_result.json"'
        },
    )


@router.get(
    "/{batch_id}/download",
    summary="Download all batch results as ZIP",
)
async def download_batch_zip(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import io
    import zipfile
    from fastapi.responses import Response

    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    items_result = await db.execute(
        select(BatchItem)
        .where(BatchItem.batch_job_id == batch_id, BatchItem.status == "completed")
        .order_by(BatchItem.position)
    )
    items = items_result.scalars().all()
    if not items:
        raise HTTPException(status_code=400, detail="No completed results to download")

    doc_ids = [item.document_id for item in items]
    docs_result = await db.execute(
        select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
    )
    doc_names = {row[0]: row[1] for row in docs_result.all()}

    storage = get_storage()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen_names = {}
        for item in items:
            item_bytes = None
            ext = "docx"
            filename = doc_names.get(item.document_id, item.document_id)
            if item.output_file and item.output_file.endswith(".docx"):
                try:
                    item_bytes = storage.load(item.output_file)
                except Exception:
                    pass
            if item_bytes is None and item.output_file:
                try:
                    item_bytes = storage.load(item.output_file)
                    ext = item.output_file.rsplit(".", 1)[-1] if "." in item.output_file else "json"
                except Exception:
                    pass
            if item_bytes is None and item.result is not None:
                try:
                    from idpkit.batch.formatter import format_result_to_docx
                    item_bytes = format_result_to_docx(batch.tool_name, item.result, filename=filename)
                    ext = "docx"
                except Exception:
                    item_bytes = json.dumps(item.result, indent=2, default=str).encode("utf-8")
                    ext = "json"
            if item_bytes is None:
                continue
            safe_name = filename.rsplit(".", 1)[0] if "." in filename else filename
            if safe_name in seen_names:
                seen_names[safe_name] += 1
                safe_name = f"{safe_name}_{seen_names[safe_name]}"
            else:
                seen_names[safe_name] = 1
            zf.writestr(f"{safe_name}_result.{ext}", item_bytes)
    buf.seek(0)

    zip_filename = f"batch_{batch_id[:8]}_results.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"'
        },
    )


@router.delete(
    "/{batch_id}",
    response_model=MessageResponse,
    summary="Delete a batch job and its output files",
)
async def delete_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    if batch.status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running batch job. Cancel it first.")

    try:
        storage = get_storage()
        output_keys = storage.list_keys(f"batch_outputs/{batch_id}/")
        for key in output_keys:
            try:
                storage.delete(key)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Failed to clean up storage for batch %s: %s", batch_id, exc)

    items_result = await db.execute(
        select(BatchItem).where(BatchItem.batch_job_id == batch_id)
    )
    for item in items_result.scalars().all():
        await db.delete(item)

    await db.delete(batch)
    await db.flush()

    return MessageResponse(detail="Batch job deleted")


@router.post(
    "/{batch_id}/cancel",
    response_model=MessageResponse,
    summary="Cancel a running batch",
)
async def cancel_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id, BatchJob.owner_id == user.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    if batch.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel batch with status '{batch.status}'")

    batch.status = "cancelled"
    await db.flush()
    return MessageResponse(detail="Batch job cancellation requested")
