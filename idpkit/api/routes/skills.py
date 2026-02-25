"""Skills API routes — CRUD for user-uploaded agent skills."""

import logging
import re
from typing import List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.api.deps import get_current_user, get_db, get_llm
from idpkit.core.llm import LLMClient
from idpkit.db.models import Skill, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillCreate(BaseModel):
    skill_content: str
    scripts: Optional[List[dict]] = None


class SkillUpdate(BaseModel):
    skill_content: Optional[str] = None
    scripts: Optional[List[dict]] = None
    is_active: Optional[bool] = None


class SkillGenerateRequest(BaseModel):
    prompt: str


def _parse_frontmatter(content: str) -> dict:
    content = content.strip()
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def _validate_skill_name(name: str) -> str:
    if not name:
        raise HTTPException(400, "Skill name is required in SKILL.md frontmatter (name field)")
    if len(name) > 64:
        raise HTTPException(400, "Skill name must be 64 characters or fewer")
    if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', name):
        raise HTTPException(400, "Skill name must be lowercase letters, numbers, and hyphens only")
    if '--' in name:
        raise HTTPException(400, "Skill name must not contain consecutive hyphens")
    return name


@router.post("", summary="Create a new skill")
async def create_skill(
    body: SkillCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    frontmatter = _parse_frontmatter(body.skill_content)
    name = _validate_skill_name(frontmatter.get("name", ""))
    description = frontmatter.get("description", "")

    existing = (await db.execute(
        select(Skill).where(Skill.owner_id == user.id, Skill.name == name)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"Skill '{name}' already exists. Use PUT to update.")

    skill = Skill(
        owner_id=user.id,
        name=name,
        description=description[:1024] if description else None,
        skill_content=body.skill_content,
        scripts=body.scripts,
        is_active=1,
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)

    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "is_active": bool(skill.is_active),
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
    }


@router.get("/library", summary="List pre-built skills available for installation")
async def list_library_skills(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from idpkit.agent.skill_library import SKILL_LIBRARY

    installed = (await db.execute(
        select(Skill.name).where(Skill.owner_id == user.id)
    )).scalars().all()
    installed_set = set(installed)

    return [
        {
            "id": s["id"],
            "category": s["category"],
            "icon": s["icon"],
            "name": _parse_frontmatter(s["skill_content"]).get("name", s["id"]),
            "description": _parse_frontmatter(s["skill_content"]).get("description", ""),
            "installed": s["id"] in installed_set,
            "skill_content": s["skill_content"],
        }
        for s in SKILL_LIBRARY
    ]


class LibraryInstallRequest(BaseModel):
    skill_id: str


@router.post("/library/install", summary="Install a pre-built skill from the library")
async def install_library_skill(
    body: LibraryInstallRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from idpkit.agent.skill_library import SKILL_LIBRARY

    lib_skill = next((s for s in SKILL_LIBRARY if s["id"] == body.skill_id), None)
    if not lib_skill:
        raise HTTPException(404, f"Library skill '{body.skill_id}' not found")

    frontmatter = _parse_frontmatter(lib_skill["skill_content"])
    name = frontmatter.get("name", body.skill_id)

    existing = (await db.execute(
        select(Skill).where(Skill.owner_id == user.id, Skill.name == name)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"Skill '{name}' is already installed")

    skill = Skill(
        owner_id=user.id,
        name=name,
        description=frontmatter.get("description", "")[:1024],
        skill_content=lib_skill["skill_content"],
        is_active=1,
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)

    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "is_active": bool(skill.is_active),
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
    }


_SKILL_GEN_PROMPT = """\
You are a skill specification writer. Generate a SKILL.md file following the Anthropic Agent Skills specification.

The user wants a skill that does the following:
{prompt}

Generate a complete SKILL.md file with:
1. YAML frontmatter between --- markers containing:
   - name: a lowercase-kebab-case name (letters, numbers, hyphens only)
   - description: a one-line summary of what the skill does (under 200 chars)
2. After the frontmatter, write detailed Markdown instructions that an AI assistant (IDA) should follow when this skill is activated. Include:
   - Clear step-by-step instructions
   - What to look for in documents
   - How to structure the output
   - Specific domain knowledge and terminology
   - Edge cases and best practices
   - Example output format if applicable

Output ONLY the SKILL.md content, nothing else. Do not wrap in code blocks."""


@router.post("/generate", summary="AI-generate a skill from natural language")
async def generate_skill(
    body: SkillGenerateRequest,
    user: User = Depends(get_current_user),
    llm: LLMClient = Depends(get_llm),
):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")
    if len(prompt) > 5000:
        raise HTTPException(400, "Prompt too long (max 5000 characters)")

    try:
        response = await llm.acomplete(_SKILL_GEN_PROMPT.format(prompt=prompt))
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        if not content.startswith("---"):
            raise HTTPException(500, "AI generated invalid skill format")

        return {"skill_content": content}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Skill generation failed: %s", exc)
        raise HTTPException(500, f"Skill generation failed: {exc}")


@router.get("", summary="List user's skills")
async def list_skills(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Skill)
        .where(Skill.owner_id == user.id)
        .order_by(Skill.created_at.desc())
    )).scalars().all()

    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "is_active": bool(s.is_active),
            "has_scripts": bool(s.scripts),
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in rows
    ]


@router.get("/{skill_id}", summary="Get skill details")
async def get_skill(
    skill_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    skill = (await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.owner_id == user.id)
    )).scalar_one_or_none()
    if not skill:
        raise HTTPException(404, "Skill not found")

    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "skill_content": skill.skill_content,
        "scripts": skill.scripts,
        "is_active": bool(skill.is_active),
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }


@router.put("/{skill_id}", summary="Update a skill")
async def update_skill(
    skill_id: str,
    body: SkillUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    skill = (await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.owner_id == user.id)
    )).scalar_one_or_none()
    if not skill:
        raise HTTPException(404, "Skill not found")

    if body.skill_content is not None:
        frontmatter = _parse_frontmatter(body.skill_content)
        name = _validate_skill_name(frontmatter.get("name", ""))
        description = frontmatter.get("description", "")

        if name != skill.name:
            conflict = (await db.execute(
                select(Skill).where(Skill.owner_id == user.id, Skill.name == name)
            )).scalar_one_or_none()
            if conflict:
                raise HTTPException(409, f"Skill '{name}' already exists")

        skill.name = name
        skill.description = description[:1024] if description else None
        skill.skill_content = body.skill_content

    if body.scripts is not None:
        skill.scripts = body.scripts

    if body.is_active is not None:
        skill.is_active = 1 if body.is_active else 0

    await db.commit()
    await db.refresh(skill)

    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "is_active": bool(skill.is_active),
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }


@router.delete("/{skill_id}", summary="Delete a skill")
async def delete_skill(
    skill_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    skill = (await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.owner_id == user.id)
    )).scalar_one_or_none()
    if not skill:
        raise HTTPException(404, "Skill not found")

    await db.delete(skill)
    await db.commit()
    return {"deleted": True, "id": skill_id, "name": skill.name}


@router.patch("/{skill_id}/toggle", summary="Toggle skill active/inactive")
async def toggle_skill(
    skill_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    skill = (await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.owner_id == user.id)
    )).scalar_one_or_none()
    if not skill:
        raise HTTPException(404, "Skill not found")

    skill.is_active = 0 if skill.is_active else 1
    await db.commit()
    await db.refresh(skill)

    return {
        "id": skill.id,
        "name": skill.name,
        "is_active": bool(skill.is_active),
    }
