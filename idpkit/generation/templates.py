"""Template management — analyze, list, save, and retrieve document templates."""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from idpkit.db.models import Template

logger = logging.getLogger(__name__)


async def analyze_template(content: str, llm: LLMClient) -> dict:
    """Identify placeholders and fields in a template using the LLM.

    Parameters
    ----------
    content:
        The raw template content (markdown, plain text, etc.).
    llm:
        An :class:`LLMClient` instance for making LLM calls.

    Returns
    -------
    dict
        A dict with keys ``"placeholders"`` (list of identified field names)
        and ``"description"`` (short summary of the template).
    """
    prompt = (
        "Analyze the following document template. Identify all placeholders, "
        "variable fields, and sections that need to be filled in. Return a JSON "
        "object with two keys:\n"
        '  "placeholders": a list of objects, each with "name" (string) and '
        '"description" (string) describing what value should go there.\n'
        '  "description": a one-sentence summary of this template\'s purpose.\n\n'
        "Template content:\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Respond with valid JSON only."
    )

    response = await llm.acomplete(prompt)
    raw = response.content.strip()

    # Attempt to parse JSON from the response.
    import json
    try:
        # Handle markdown code blocks wrapping the JSON.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response for template analysis")
        result = {
            "placeholders": [],
            "description": raw[:200],
        }

    return result


async def list_templates(db: AsyncSession, owner_id: Optional[str] = None) -> list[dict]:
    """Return all templates, optionally filtered by owner.

    Parameters
    ----------
    db:
        An async database session.
    owner_id:
        If provided, only return templates owned by this user.

    Returns
    -------
    list[dict]
        A list of template summary dicts.
    """
    stmt = select(Template)
    if owner_id:
        stmt = stmt.where(Template.owner_id == owner_id)
    stmt = stmt.order_by(Template.created_at.desc())

    result = await db.execute(stmt)
    templates = result.scalars().all()

    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "format": t.format,
            "created_at": str(t.created_at),
        }
        for t in templates
    ]


async def save_template(
    name: str,
    content: str,
    format: str,
    db: AsyncSession,
    owner_id: str,
    description: Optional[str] = None,
    schema_json: Optional[dict] = None,
) -> dict:
    """Save a new template to the database.

    Parameters
    ----------
    name:
        Display name for the template.
    content:
        The template content (markdown, text, etc.).
    format:
        Output format this template targets (``"docx"``, ``"md"``).
    db:
        An async database session.
    owner_id:
        The ID of the user who owns this template.
    description:
        Optional description.
    schema_json:
        Optional JSON schema describing the template fields.

    Returns
    -------
    dict
        The saved template as a dict.
    """
    template = Template(
        name=name,
        content=content,
        format=format,
        description=description,
        schema_json=schema_json,
        owner_id=owner_id,
    )
    db.add(template)
    await db.flush()
    await db.refresh(template)

    logger.info("Template saved: %s (id=%s)", name, template.id)

    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "format": template.format,
        "content": template.content,
        "created_at": str(template.created_at),
    }


async def get_template(template_id: str, db: AsyncSession) -> dict:
    """Retrieve a template by ID.

    Parameters
    ----------
    template_id:
        The unique template identifier.
    db:
        An async database session.

    Returns
    -------
    dict
        The template as a dict, or an empty dict if not found.
    """
    result = await db.execute(select(Template).where(Template.id == template_id))
    t = result.scalar_one_or_none()

    if not t:
        return {}

    return {
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "format": t.format,
        "content": t.content,
        "schema_json": t.schema_json,
        "file_path": t.file_path,
        "created_at": str(t.created_at),
    }
