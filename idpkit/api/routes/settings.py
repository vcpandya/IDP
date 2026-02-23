"""IDP Kit Settings API — providers, models, and prompt management."""

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import Prompt, User
from idpkit.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# Provider / model definitions
# ---------------------------------------------------------------------------

PROVIDERS = [
    {
        "id": "openai",
        "name": "OpenAI",
        "icon": "fa-brain",
        "env_keys": ["OPENAI_API_KEY", "CHATGPT_API_KEY"],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "icon": "fa-robot",
        "env_keys": ["ANTHROPIC_API_KEY"],
    },
    {
        "id": "google",
        "name": "Google",
        "icon": "fa-google",
        "env_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "icon": "fa-route",
        "env_keys": ["OPENROUTER_API_KEY"],
    },
    {
        "id": "ollama",
        "name": "Ollama (Local)",
        "icon": "fa-server",
        "env_keys": [],  # detected via HTTP probe
    },
]

CURATED_MODELS = {
    "anthropic": [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "anthropic"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "provider": "anthropic"},
        {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "anthropic"},
    ],
    "google": [
        {"id": "gemini/gemini-2.5-flash-preview-04-17", "name": "Gemini 2.5 Flash Preview", "provider": "google"},
        {"id": "gemini/gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro Preview", "provider": "google"},
        {"id": "gemini/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "google"},
        {"id": "gemini/gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google"},
    ],
}


def _env_key_for(provider_id: str) -> Optional[str]:
    """Return the first set environment key for a provider, or None."""
    info = next((p for p in PROVIDERS if p["id"] == provider_id), None)
    if not info:
        return None
    for key in info["env_keys"]:
        if os.getenv(key):
            return key
    return None


async def _ollama_available() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{base}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GET /api/settings/providers
# ---------------------------------------------------------------------------

@router.get("/providers")
async def list_providers(user: User = Depends(get_current_user)):
    """Return available LLM providers with configuration status."""
    results = []
    for p in PROVIDERS:
        if p["id"] == "ollama":
            configured = await _ollama_available()
        else:
            configured = _env_key_for(p["id"]) is not None
        results.append({
            "id": p["id"],
            "name": p["name"],
            "icon": p["icon"],
            "configured": configured,
        })
    return results


# ---------------------------------------------------------------------------
# GET /api/settings/models?provider=openai
# ---------------------------------------------------------------------------

async def _fetch_openai_models() -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            for m in data.get("data", []):
                mid = m["id"]
                # Filter to GPT / chat models — skip embeddings, tts, dall-e, whisper, etc.
                if any(mid.startswith(pref) for pref in ("gpt-", "o1", "o3", "o4", "chatgpt-")):
                    models.append({"id": mid, "name": mid, "provider": "openai"})
            models.sort(key=lambda x: x["name"])
            return models
    except Exception as exc:
        logger.warning("Failed to fetch OpenAI models: %s", exc)
        return []


async def _fetch_openrouter_models() -> list[dict]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            for m in data.get("data", []):
                models.append({
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "provider": "openrouter",
                })
            models.sort(key=lambda x: x["name"])
            return models
    except Exception as exc:
        logger.warning("Failed to fetch OpenRouter models: %s", exc)
        return []


async def _fetch_ollama_models() -> list[dict]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                models.append({
                    "id": f"ollama/{name}",
                    "name": name,
                    "provider": "ollama",
                })
            models.sort(key=lambda x: x["name"])
            return models
    except Exception:
        return []


@router.get("/models")
async def list_models(
    provider: str = Query(..., description="Provider ID"),
    user: User = Depends(get_current_user),
):
    """Fetch available models for a given provider."""
    if provider == "openai":
        models = await _fetch_openai_models()
        if not models:
            # Fallback curated list
            models = [
                {"id": "gpt-4o", "name": "gpt-4o", "provider": "openai"},
                {"id": "gpt-4o-mini", "name": "gpt-4o-mini", "provider": "openai"},
                {"id": "gpt-4.1", "name": "gpt-4.1", "provider": "openai"},
                {"id": "gpt-4.1-mini", "name": "gpt-4.1-mini", "provider": "openai"},
                {"id": "gpt-4.1-nano", "name": "gpt-4.1-nano", "provider": "openai"},
                {"id": "o4-mini", "name": "o4-mini", "provider": "openai"},
            ]
    elif provider == "anthropic":
        models = CURATED_MODELS["anthropic"]
    elif provider == "google":
        models = CURATED_MODELS["google"]
    elif provider == "openrouter":
        models = await _fetch_openrouter_models()
        if not models:
            models = [
                {"id": "openrouter/auto", "name": "Auto (best available)", "provider": "openrouter"},
            ]
    elif provider == "ollama":
        models = await _fetch_ollama_models()
        if not models:
            models = [
                {"id": "ollama/llama3", "name": "llama3 (default)", "provider": "ollama"},
            ]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    return models


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------

class PromptCreate(BaseModel):
    name: str = Field(..., max_length=200)
    content: str
    category: Optional[str] = None


@router.get("/prompts")
async def list_prompts(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List saved prompts for the current user."""
    result = await db.execute(
        select(Prompt)
        .where(Prompt.owner_id == user.id)
        .order_by(Prompt.updated_at.desc())
    )
    prompts = result.scalars().all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "content": p.content,
            "category": p.category,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in prompts
    ]


@router.post("/prompts", status_code=201)
async def create_prompt(
    body: PromptCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a new prompt."""
    prompt = Prompt(
        name=body.name,
        content=body.content,
        category=body.category,
        owner_id=user.id,
    )
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return {
        "id": prompt.id,
        "name": prompt.name,
        "content": prompt.content,
        "category": prompt.category,
    }


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a saved prompt."""
    result = await db.execute(
        select(Prompt).where(Prompt.id == prompt_id, Prompt.owner_id == user.id)
    )
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    await db.delete(prompt)
    await db.commit()
    return {"ok": True}
