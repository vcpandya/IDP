"""Skills loader and matcher for the IDA agent.

Loads user's active skills from the database, builds a skills summary
for the system prompt, and provides a tool for the agent to use skills.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.models import Skill

logger = logging.getLogger(__name__)


async def load_active_skills(db: AsyncSession, user_id: str) -> list[dict]:
    rows = (await db.execute(
        select(Skill)
        .where(Skill.owner_id == user_id, Skill.is_active == 1)
        .order_by(Skill.name)
    )).scalars().all()

    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description or "",
        }
        for s in rows
    ]


def build_skills_prompt_section(skills: list[dict]) -> str:
    if not skills:
        return ""

    lines = [
        "\n\n### User Skills (Custom Extensions)",
        "The user has uploaded custom skills that extend your capabilities.",
        "When a task matches a skill's description, call `use_skill` to load its full instructions.",
        "Available skills:",
    ]
    for s in skills:
        lines.append(f"- **{s['name']}**: {s['description']}")

    lines.append(
        "\nTo use a skill, call the `use_skill` tool with the skill name. "
        "This will return the full SKILL.md content with detailed instructions to follow. "
        "If a skill includes scripts, you can execute them using `execute_python`."
    )
    return "\n".join(lines)


async def get_skill_content(db: AsyncSession, user_id: str, skill_name: str) -> dict[str, Any]:
    skill = (await db.execute(
        select(Skill).where(
            Skill.owner_id == user_id,
            Skill.name == skill_name,
            Skill.is_active == 1,
        )
    )).scalar_one_or_none()

    if not skill:
        return {"error": f"Skill '{skill_name}' not found or is inactive."}

    result: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.skill_content,
    }

    if skill.scripts:
        result["scripts"] = skill.scripts

    return result
