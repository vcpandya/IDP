"""Document verification engine using multimodal LLM capabilities."""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
from typing import Any

logger = logging.getLogger(__name__)

VISION_MODEL = os.getenv("IDP_VISION_MODEL", "gpt-4o")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
MAX_PDF_PAGES_AS_IMAGES = 5


def _file_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext


def _image_to_base64_block(data: bytes, mime: str) -> dict:
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _pdf_pages_to_images(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES_AS_IMAGES) -> list[bytes]:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available; cannot render PDF pages as images")
        return []

    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = min(len(doc), max_pages)
        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=150)
            images.append(pix.tobytes("png"))
        doc.close()
    except Exception as exc:
        logger.error("Failed to render PDF pages: %s", exc)
    return images


def _pdf_extract_text(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES_AS_IMAGES) -> str:
    try:
        import fitz
    except ImportError:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for i in range(min(len(doc), max_pages)):
            texts.append(doc[i].get_text())
        doc.close()
        return "\n\n".join(texts)
    except Exception as exc:
        logger.error("Failed to extract PDF text: %s", exc)
        return ""


def _extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.warning("DOCX text extraction failed: %s", exc)
        return ""


def _extract_text_fallback(file_bytes: bytes, filename: str) -> str:
    ext = _file_extension(filename)
    if ext in (".docx", ".doc"):
        return _extract_text_from_docx(file_bytes)
    if ext in (".txt", ".md", ".markdown", ".csv"):
        try:
            return file_bytes.decode("utf-8", errors="replace")[:20000]
        except Exception:
            return ""
    if ext in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(file_bytes, "html.parser").get_text(separator="\n")[:20000]
        except Exception:
            return ""
    return ""


def _build_verification_prompt(expectations: list[str]) -> str:
    exp_list = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(expectations))
    return f"""You are a document verification expert. Analyze the provided document and verify whether it matches each of the following expected criteria:

{exp_list}

For EACH expectation, determine if the document satisfies it.

Respond with ONLY valid JSON in this exact format:
{{
  "status": "ok" or "not_ok",
  "matches": [
    {{
      "expected": "<the expectation text>",
      "result": "ok" or "not_ok",
      "confidence": "high" or "medium" or "low",
      "details": "<brief explanation of why it matches or doesn't>"
    }}
  ],
  "summary": "<one-sentence overall summary of the verification>"
}}

Rules:
- "status" is "ok" ONLY if ALL expectations are met; otherwise "not_ok"
- Be precise and specific in your details
- For identity documents (Aadhaar, PAN, passport, etc.), verify the document TYPE matches but do NOT extract or expose personal information
- For certificates, check if the document appears to be a legitimate certificate of the expected type
- If the document is unclear, blurry, or unreadable, say so in details and mark as "not_ok"
"""


async def verify_document(
    file_bytes: bytes,
    filename: str,
    expectations: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    from idpkit.core.llm import get_default_client

    client = get_default_client()
    used_model = model or VISION_MODEL

    ext = _file_extension(filename)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prompt_text = _build_verification_prompt(expectations)

    content_blocks: list[dict] = [{"type": "text", "text": prompt_text}]

    if ext in IMAGE_EXTENSIONS:
        content_blocks.append(_image_to_base64_block(file_bytes, mime))

    elif ext in PDF_EXTENSIONS:
        page_images = _pdf_pages_to_images(file_bytes)
        if page_images:
            for img_data in page_images:
                content_blocks.append(_image_to_base64_block(img_data, "image/png"))
        fallback_text = _pdf_extract_text(file_bytes)
        if fallback_text.strip():
            content_blocks.append({
                "type": "text",
                "text": f"\n--- Extracted text from PDF ---\n{fallback_text[:15000]}",
            })

    else:
        text = _extract_text_fallback(file_bytes, filename)
        if text.strip():
            content_blocks.append({
                "type": "text",
                "text": f"\n--- Document text content ---\n{text[:15000]}",
            })
        else:
            return {
                "filename": filename,
                "status": "not_ok",
                "matches": [
                    {
                        "expected": e,
                        "result": "not_ok",
                        "confidence": "low",
                        "details": "Could not extract any content from this file format.",
                    }
                    for e in expectations
                ],
                "summary": "Unable to read document content for verification.",
            }

    try:
        response = await client.acomplete(
            prompt=content_blocks,
            model=used_model,
            max_tokens=2000,
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        result["filename"] = filename
        return result

    except json.JSONDecodeError as exc:
        logger.error("LLM returned non-JSON for %s: %s", filename, exc)
        return {
            "filename": filename,
            "status": "not_ok",
            "matches": [
                {"expected": e, "result": "not_ok", "confidence": "low",
                 "details": "Verification inconclusive — could not parse AI response."}
                for e in expectations
            ],
            "summary": "Verification failed: could not parse LLM response.",
        }
    except Exception as exc:
        logger.error("Verification failed for %s: %s", filename, exc)
        return {
            "filename": filename,
            "status": "not_ok",
            "matches": [
                {"expected": e, "result": "not_ok", "confidence": "low",
                 "details": f"Verification error: {exc}"}
                for e in expectations
            ],
            "summary": f"Verification error: {exc}",
        }
