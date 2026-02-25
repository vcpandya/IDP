"""Document Verifier API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from idpkit.api.deps import get_current_user
from idpkit.db.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/verify", tags=["verifier"])

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_FILES = 20
ALLOWED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
    ".docx", ".doc", ".txt", ".md", ".html", ".htm", ".csv",
}


class SingleVerifyResult(BaseModel):
    filename: str
    status: str
    matches: list[dict] = Field(default_factory=list)
    summary: str = ""


def _validate_file(file: UploadFile) -> str:
    import os
    _, ext = os.path.splitext((file.filename or "").lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


def _parse_expectations(expectations_json: str) -> list[str]:
    try:
        parsed = json.loads(expectations_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in 'expectations' field.",
        )
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(e, str) for e in parsed):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'expectations' must be a JSON string or array of strings.",
        )
    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one expectation is required.",
        )
    return parsed


@router.post("/single", response_model=SingleVerifyResult)
async def verify_single(
    file: UploadFile = File(...),
    expectations: str = Form(...),
    model: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    _validate_file(file)
    exps = _parse_expectations(expectations)

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    from idpkit.verifier.engine import verify_document
    result = await verify_document(file_bytes, file.filename or "unknown", exps, model)
    return SingleVerifyResult(**result)


@router.post("")
async def verify_documents(
    files: list[UploadFile] = File(...),
    expectations: str = Form(...),
    model: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum is {MAX_FILES}.",
        )
    for f in files:
        _validate_file(f)
    exps = _parse_expectations(expectations)

    file_data: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{f.filename}' too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
            )
        file_data.append((f.filename or "unknown", data))

    if len(file_data) == 1:
        from idpkit.verifier.engine import verify_document
        result = await verify_document(file_data[0][1], file_data[0][0], exps, model)
        return result

    async def _sse_generator():
        from idpkit.verifier.engine import verify_document

        ok_count = 0
        total = len(file_data)

        for idx, (fname, fbytes) in enumerate(file_data):
            try:
                result = await verify_document(fbytes, fname, exps, model)
                result["index"] = idx
                if result.get("status") == "ok":
                    ok_count += 1
            except Exception as exc:
                logger.error("Verification error for %s: %s", fname, exc)
                result = {
                    "index": idx,
                    "filename": fname,
                    "status": "not_ok",
                    "matches": [
                        {"expected": e, "result": "not_ok", "confidence": "low",
                         "details": f"Verification error: {exc}"}
                        for e in exps
                    ],
                    "summary": f"Error: {exc}",
                }

            yield f"data: {json.dumps(result)}\n\n"

        summary = {
            "done": True,
            "total": total,
            "ok_count": ok_count,
            "not_ok_count": total - ok_count,
        }
        yield f"data: {json.dumps(summary)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
