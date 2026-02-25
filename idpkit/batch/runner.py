"""Batch job runner — executes batch processing jobs in the background.

Follows the same pattern as ``idpkit/jobs/manager.py``: creates its own
session factory so it never shares a session with the originating HTTP request.
Uses ``asyncio.Semaphore`` for concurrency control.
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timezone

from idpkit.core.llm import LLMClient, get_default_client

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_session_factory(db_url: str | None):
    """Return an async session factory (mirrors jobs/manager.py)."""
    if db_url is None:
        from idpkit.db.session import async_session
        return async_session

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    _engine = create_async_engine(db_url, echo=False)
    return async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def _extract_text_from_tree(tree_index: dict | list | None) -> str:
    """Recursively extract all text from a tree index structure."""
    if tree_index is None:
        return ""

    parts: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            if node.get("title"):
                parts.append(node["title"])
            if node.get("text"):
                parts.append(node["text"])
            if node.get("summary"):
                parts.append(node["summary"])
            for child in node.get("nodes", []):
                _walk(child)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(tree_index)
    return "\n\n".join(parts)


async def _generate_output_schema(
    *,
    prompt: str,
    reference_texts: list[str],
    llm: LLMClient,
) -> dict | None:
    """Pass 1: Ask the LLM to generate a JSON schema based on the prompt and reference documents.

    Returns a JSON schema dict, or None if generation fails.
    """
    system_msg = (
        "You are a schema architect. Based on the user's processing instructions and any "
        "reference documents provided, generate a JSON Schema that defines the ideal structured "
        "output format. The schema should capture all the fields, types, and structure that the "
        "instructions imply.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON Schema (no markdown, no explanation)\n"
        "- Use descriptive field names\n"
        "- Include 'type', 'properties', and 'required' at minimum\n"
        "- Use appropriate types: string, number, integer, boolean, array, object\n"
        "- Add 'description' to each property to clarify what it captures\n"
        "- The schema should be reusable across multiple similar documents"
    )

    user_content = f"Processing Instructions:\n{prompt}\n"
    if reference_texts:
        user_content += "\nReference Documents (for context on expected data):\n"
        for i, text in enumerate(reference_texts, 1):
            preview = text[:3000]
            user_content += f"\n--- Reference {i} ---\n{preview}\n"

    user_content += (
        "\nBased on these instructions and reference materials, generate the JSON Schema "
        "for the structured output format."
    )

    try:
        response = await llm.acomplete(
            user_content,
            chat_history=[{"role": "system", "content": system_msg}],
        )
        import json as _json
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        schema = _json.loads(raw)
        if isinstance(schema, dict) and "properties" in schema:
            logger.info("Generated output schema with %d properties", len(schema.get("properties", {})))
            return schema
        logger.warning("Generated schema missing 'properties' key, discarding")
        return None
    except Exception as exc:
        logger.warning("Failed to generate output schema: %s", exc)
        return None


async def _load_reference_doc_texts(
    reference_doc_ids: list[str],
    session_factory,
) -> list[str]:
    """Load text content from reference documents."""
    from sqlalchemy import select
    from idpkit.db.models import Document

    texts = []
    async with session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id.in_(reference_doc_ids))
        )
        for doc in result.scalars().all():
            text = _extract_text_from_tree(doc.tree_index)
            if text:
                texts.append(f"[{doc.filename}]\n{text}")
    return texts


async def run_batch_job(batch_id: str, db_url: str | None = None) -> dict:
    """Execute a batch processing job.

    Parameters
    ----------
    batch_id:
        Primary key of the BatchJob row.
    db_url:
        Optional database URL. When None, uses the app-wide session.

    Returns
    -------
    dict
        Summary of batch results.
    """
    from sqlalchemy import select
    from idpkit.db.models import BatchJob, BatchItem, Document, ProcessingTemplate

    session_factory = await _get_session_factory(db_url)

    # ------------------------------------------------------------------
    # 1. Load batch job and mark as running
    # ------------------------------------------------------------------
    async with session_factory() as session:
        result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
        batch = result.scalar_one_or_none()
        if not batch:
            logger.error("BatchJob %s not found", batch_id)
            return {"error": "Batch job not found"}

        batch.status = "running"
        batch.started_at = _utcnow()
        await session.commit()

        # Capture values we need outside the session
        tool_name = batch.tool_name
        template_id = batch.template_id
        batch_prompt = batch.prompt
        batch_options = batch.options or {}
        batch_model = batch.model
        concurrency = batch.concurrency or 3
        owner_id = batch.owner_id
        reference_doc_ids = batch.reference_doc_ids or []

    if not batch_model and owner_id:
        from idpkit.db.models import User
        async with session_factory() as session:
            user_result = await session.execute(select(User).where(User.id == owner_id))
            owner = user_result.scalar_one_or_none()
            if owner and owner.default_model:
                batch_model = owner.default_model

    # ------------------------------------------------------------------
    # 2. Load template if referenced
    # ------------------------------------------------------------------
    template_prompt = None
    template_options = {}
    template_output_format = "json"
    template_output_template = None
    template_reference_content = None

    if template_id:
        async with session_factory() as session:
            result = await session.execute(
                select(ProcessingTemplate).where(ProcessingTemplate.id == template_id)
            )
            tmpl = result.scalar_one_or_none()
            if tmpl:
                template_prompt = tmpl.system_prompt
                template_options = tmpl.tool_options or {}
                template_output_format = tmpl.output_format or "json"
                template_output_template = tmpl.output_template
                template_reference_content = tmpl.reference_content
                if not batch_model and tmpl.model:
                    batch_model = tmpl.model

    # Merge: template values as defaults, batch body overrides
    merged_prompt = batch_prompt or template_prompt or ""
    merged_options = {**template_options, **batch_options}

    # ------------------------------------------------------------------
    # 2b. Load reference documents and generate schema (two-pass)
    # ------------------------------------------------------------------
    generated_schema = None
    reference_doc_texts: list[str] = []

    if reference_doc_ids:
        reference_doc_texts = await _load_reference_doc_texts(reference_doc_ids, session_factory)

    if tool_name == "custom" and not template_output_template and merged_prompt:
        if reference_doc_texts or merged_prompt:
            temp_llm = get_default_client()
            if batch_model:
                temp_llm = LLMClient(
                    default_model=batch_model,
                    api_base=temp_llm.api_base,
                )
            logger.info(
                "Batch %s: Running Pass 1 — generating output schema from prompt + %d reference docs",
                batch_id, len(reference_doc_texts),
            )
            generated_schema = await _generate_output_schema(
                prompt=merged_prompt,
                reference_texts=reference_doc_texts,
                llm=temp_llm,
            )
            if generated_schema:
                async with session_factory() as session:
                    result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
                    batch_row = result.scalar_one_or_none()
                    if batch_row:
                        batch_row.generated_schema = generated_schema
                        await session.commit()
                logger.info("Batch %s: Schema generated successfully, proceeding to Pass 2", batch_id)
            else:
                logger.info("Batch %s: Schema generation returned None, proceeding without structured output", batch_id)

    # ------------------------------------------------------------------
    # 3. Get tool instance (or None for "custom" mode)
    # ------------------------------------------------------------------
    tool_instance = None
    if tool_name != "custom":
        from idpkit.tools import TOOL_REGISTRY
        tool_instance = TOOL_REGISTRY.get(tool_name)
        if not tool_instance:
            async with session_factory() as session:
                result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
                batch = result.scalar_one_or_none()
                if batch:
                    batch.status = "failed"
                    batch.error = f"Tool '{tool_name}' not found in registry"
                    batch.completed_at = _utcnow()
                    await session.commit()
            return {"error": f"Tool '{tool_name}' not found"}

    # ------------------------------------------------------------------
    # 4. Build LLM client
    # ------------------------------------------------------------------
    llm = get_default_client()
    if batch_model:
        llm = LLMClient(
            default_model=batch_model,
            api_base=llm.api_base,
        )

    # ------------------------------------------------------------------
    # 5. Load all batch items
    # ------------------------------------------------------------------
    async with session_factory() as session:
        result = await session.execute(
            select(BatchItem)
            .where(BatchItem.batch_job_id == batch_id)
            .order_by(BatchItem.position)
        )
        items_data = [
            {"id": item.id, "document_id": item.document_id}
            for item in result.scalars().all()
        ]

    # ------------------------------------------------------------------
    # 6. Process items with concurrency control
    # ------------------------------------------------------------------
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    failed_count = 0
    lock = asyncio.Lock()

    async def process_item(item_id: str, document_id: str):
        nonlocal completed_count, failed_count

        async with semaphore:
            # Check for cancellation
            async with session_factory() as session:
                result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
                batch_check = result.scalar_one_or_none()
                if batch_check and batch_check.status == "cancelled":
                    # Skip this item
                    result2 = await session.execute(
                        select(BatchItem).where(BatchItem.id == item_id)
                    )
                    item = result2.scalar_one_or_none()
                    if item and item.status == "pending":
                        item.status = "skipped"
                        await session.commit()
                    return

            # Mark item as running
            async with session_factory() as session:
                result = await session.execute(
                    select(BatchItem).where(BatchItem.id == item_id)
                )
                item = result.scalar_one_or_none()
                if item:
                    item.status = "running"
                    item.started_at = _utcnow()
                    await session.commit()

            try:
                # Load document
                async with session_factory() as session:
                    result = await session.execute(
                        select(Document).where(Document.id == document_id)
                    )
                    doc = result.scalar_one_or_none()
                    if not doc:
                        raise ValueError(f"Document {document_id} not found")

                    # Auto-index if needed
                    auto_index = merged_options.pop("_auto_index", False) if "_auto_index" in merged_options else False
                    if auto_index and doc.status != "indexed":
                        await _auto_index_document(doc, session_factory)
                        # Reload to get updated tree_index
                        result = await session.execute(
                            select(Document).where(Document.id == document_id)
                        )
                        doc = result.scalar_one_or_none()

                    doc_tree_index = doc.tree_index
                    doc_status = doc.status
                    doc_filename = doc.filename

                # Execute tool or custom LLM
                if tool_name == "custom":
                    ref_content = template_reference_content
                    if not ref_content and reference_doc_texts:
                        ref_content = "\n\n---\n\n".join(reference_doc_texts)
                    item_result = await _run_custom_llm(
                        doc_tree_index=doc_tree_index,
                        doc_filename=doc_filename,
                        prompt=merged_prompt,
                        reference_content=ref_content,
                        llm=llm,
                        output_schema=generated_schema,
                    )
                else:
                    # Inject prompt into options if provided
                    exec_options = dict(merged_options)
                    if merged_prompt:
                        exec_options["prompt"] = merged_prompt

                    async with session_factory() as session:
                        item_result = await tool_instance.execute(
                            document_id=document_id,
                            options=exec_options,
                            llm=llm,
                            db=session,
                        )

                # Serialize result
                result_data = None
                if hasattr(item_result, "data"):
                    result_data = item_result.data
                elif isinstance(item_result, dict):
                    result_data = item_result

                item_status = "completed"
                item_error = None
                if hasattr(item_result, "status") and item_result.status == "error":
                    item_status = "failed"
                    item_error = getattr(item_result, "error", "Tool returned error status")

                # Persist result to storage as DOCX and JSON
                output_file_key = None
                if item_status == "completed" and result_data is not None:
                    try:
                        from idpkit.api.deps import get_storage as _get_storage
                        from idpkit.batch.formatter import format_result_to_docx
                        storage = _get_storage()
                        original_text = _extract_text_from_tree(doc_tree_index)
                        output_file_key = f"batch_outputs/{batch_id}/{item_id}.docx"
                        docx_bytes = format_result_to_docx(
                            tool_name, result_data,
                            filename=doc_filename,
                            original_text=original_text if original_text else None,
                        )
                        storage.save(output_file_key, docx_bytes)
                        json_key = f"batch_outputs/{batch_id}/{item_id}.json"
                        storage.save(json_key, json.dumps(result_data, indent=2, default=str).encode("utf-8"))
                    except Exception as exc:
                        logger.warning("Failed to persist batch item %s output to storage: %s", item_id, exc)
                        output_file_key = None

                # Update item
                async with session_factory() as session:
                    result = await session.execute(
                        select(BatchItem).where(BatchItem.id == item_id)
                    )
                    item = result.scalar_one_or_none()
                    if item:
                        item.status = item_status
                        item.result = result_data
                        item.error = item_error
                        item.output_file = output_file_key or getattr(item_result, "output_file", None)
                        item.completed_at = _utcnow()
                        await session.commit()

                async with lock:
                    if item_status == "completed":
                        completed_count += 1
                    else:
                        failed_count += 1

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error("Batch item %s failed: %s", item_id, error_msg)

                async with session_factory() as session:
                    result = await session.execute(
                        select(BatchItem).where(BatchItem.id == item_id)
                    )
                    item = result.scalar_one_or_none()
                    if item:
                        item.status = "failed"
                        item.error = error_msg
                        item.completed_at = _utcnow()
                        await session.commit()

                async with lock:
                    failed_count += 1

            # Update batch progress
            async with lock:
                total_done = completed_count + failed_count
            async with session_factory() as session:
                result = await session.execute(
                    select(BatchJob).where(BatchJob.id == batch_id)
                )
                batch_row = result.scalar_one_or_none()
                if batch_row:
                    batch_row.completed_items = completed_count
                    batch_row.failed_items = failed_count
                    total = batch_row.total_items or len(items_data)
                    batch_row.progress = int((total_done / total) * 100) if total > 0 else 0
                    await session.commit()

    # Run all items concurrently (bounded by semaphore)
    try:
        await asyncio.gather(
            *[process_item(item["id"], item["document_id"]) for item in items_data],
            return_exceptions=True,
        )
    except Exception as exc:
        logger.error("Batch job %s gather failed: %s", batch_id, exc)

    # ------------------------------------------------------------------
    # 7. Finalize batch
    # ------------------------------------------------------------------
    async with session_factory() as session:
        result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
        batch = result.scalar_one_or_none()
        if batch:
            if batch.status != "cancelled":
                batch.status = "completed" if failed_count == 0 else (
                    "failed" if completed_count == 0 else "completed"
                )
            batch.progress = 100
            batch.completed_items = completed_count
            batch.failed_items = failed_count
            batch.completed_at = _utcnow()
            batch.result_summary = {
                "total": len(items_data),
                "completed": completed_count,
                "failed": failed_count,
            }
            await session.commit()

    logger.info(
        "Batch %s finished: %d completed, %d failed out of %d",
        batch_id, completed_count, failed_count, len(items_data),
    )
    return {"total": len(items_data), "completed": completed_count, "failed": failed_count}


async def _run_custom_llm(
    *,
    doc_tree_index,
    doc_filename: str,
    prompt: str,
    reference_content: str | None,
    llm: LLMClient,
    output_schema: dict | None = None,
) -> dict:
    """Run raw LLM processing (custom mode).

    If ``output_schema`` is provided (from Pass 1), instructs the LLM to
    produce structured JSON matching the schema.
    """
    doc_text = _extract_text_from_tree(doc_tree_index)
    if not doc_text:
        doc_text = "(No indexed content available for this document)"

    messages = []
    system_content = prompt or ""
    if output_schema:
        import json as _json
        schema_str = _json.dumps(output_schema, indent=2)
        system_content += (
            "\n\nIMPORTANT: You MUST return your response as valid JSON that conforms "
            "to the following JSON Schema. Do NOT include any text outside the JSON object.\n\n"
            f"JSON Schema:\n{schema_str}"
        )
    if system_content:
        messages.append({"role": "system", "content": system_content})

    user_content = f"Document: {doc_filename}\n\n"
    if reference_content:
        user_content += f"Reference/Instructions:\n{reference_content}\n\n"
    user_content += f"Document Content:\n{doc_text}"

    llm_kwargs = {}
    if output_schema:
        llm_kwargs["response_format"] = {"type": "json_object"}

    response = await llm.acomplete(
        user_content,
        chat_history=messages if messages else None,
        **llm_kwargs,
    )

    result_data: dict | str = response.content
    if output_schema:
        try:
            import json as _json
            result_data = _json.loads(response.content)
        except Exception:
            result_data = response.content

    return {"status": "success", "data": {"response": result_data}}


async def _auto_index_document(doc, session_factory):
    """Index a document inline (for upload-and-process flow)."""
    import os
    from idpkit.indexing import get_indexer
    from idpkit.api.deps import get_storage

    storage = get_storage()

    if not doc.file_path or not storage.exists(doc.file_path):
        raise ValueError(f"Document file not found: {doc.file_path}")

    ext = os.path.splitext(doc.filename)[1].lower()
    indexer = get_indexer(ext)

    # Get the absolute path for the indexer
    abs_path = storage.get_path(doc.file_path)
    result = await indexer.build_index(abs_path)

    # Save the index
    from sqlalchemy import select
    from idpkit.db.models import Document

    async with session_factory() as session:
        row = (await session.execute(
            select(Document).where(Document.id == doc.id)
        )).scalar_one_or_none()
        if row:
            row.tree_index = result.get("structure")
            if "doc_description" in result:
                row.description = result["doc_description"]
            row.status = "indexed"
            row.updated_at = _utcnow()
            await session.commit()
