"""IDP Kit Admin API routes — user management, role management, rate limits."""

import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User, UserRole, SystemSetting
from idpkit.api.deps import require_admin, require_superadmin, get_rate_limit, update_rate_limit_cache, DEFAULT_RATE_LIMITS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class AdminUserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: int
    created_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    users: List[AdminUserResponse]
    total: int


class MessageResponse(BaseModel):
    detail: str


def _is_admin_role(role: str) -> bool:
    return role in (UserRole.ADMIN.value, UserRole.SUPERADMIN.value)


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all users (admin only)",
)
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).order_by(User.is_active.asc(), User.created_at.desc())
    )
    users = result.scalars().all()

    count_result = await db.execute(select(func.count(User.id)))
    total = count_result.scalar()

    return UserListResponse(users=users, total=total)


@router.post(
    "/users/{user_id}/approve",
    response_model=MessageResponse,
    summary="Approve a pending user (admin only)",
)
async def approve_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own account")

    user.is_active = 1
    db.add(user)
    await db.flush()

    logger.info("User approved by %s: %s (id=%s)", admin.username, user.username, user.id)
    return MessageResponse(detail=f"User '{user.username}' has been approved")


@router.post(
    "/users/{user_id}/deactivate",
    response_model=MessageResponse,
    summary="Deactivate a user (admin only)",
)
async def deactivate_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    if user.role == UserRole.SUPERADMIN.value:
        raise HTTPException(status_code=403, detail="Cannot deactivate the superadmin")
    if user.role == UserRole.ADMIN.value and admin.role != UserRole.SUPERADMIN.value:
        raise HTTPException(status_code=403, detail="Only superadmin can deactivate other admins")

    user.is_active = 0
    db.add(user)
    await db.flush()

    logger.info("User deactivated by %s: %s (id=%s)", admin.username, user.username, user.id)
    return MessageResponse(detail=f"User '{user.username}' has been deactivated")


@router.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    summary="Delete a user (admin only)",
)
async def delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if user.role == UserRole.SUPERADMIN.value:
        raise HTTPException(status_code=403, detail="Cannot delete the superadmin")
    if user.role == UserRole.ADMIN.value and admin.role != UserRole.SUPERADMIN.value:
        raise HTTPException(status_code=403, detail="Only superadmin can delete other admins")

    username = user.username
    await db.delete(user)
    await db.flush()

    logger.info("User deleted by %s: %s (id=%s)", admin.username, username, user_id)
    return MessageResponse(detail=f"User '{username}' has been deleted")


@router.post(
    "/users/{user_id}/promote",
    response_model=MessageResponse,
    summary="Promote a user to admin (admin only)",
)
async def promote_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own role")
    if _is_admin_role(user.role):
        raise HTTPException(status_code=400, detail="User is already an admin")

    user.role = UserRole.ADMIN.value
    db.add(user)
    await db.flush()

    logger.info("User promoted to admin by %s: %s (id=%s)", admin.username, user.username, user.id)
    return MessageResponse(detail=f"User '{user.username}' has been promoted to admin")


@router.post(
    "/users/{user_id}/demote",
    response_model=MessageResponse,
    summary="Demote an admin to regular user (superadmin only)",
)
async def demote_user(
    user_id: str,
    admin: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot demote yourself")
    if user.role == UserRole.SUPERADMIN.value:
        raise HTTPException(status_code=403, detail="Cannot demote the superadmin")
    if user.role != UserRole.ADMIN.value:
        raise HTTPException(status_code=400, detail="User is not an admin")

    user.role = UserRole.USER.value
    db.add(user)
    await db.flush()

    logger.info("User demoted by %s: %s (id=%s)", admin.username, user.username, user.id)
    return MessageResponse(detail=f"User '{user.username}' has been demoted to regular user")


class RateLimitsResponse(BaseModel):
    agent_chat: str
    batch_create: str


class RateLimitsUpdate(BaseModel):
    agent_chat: str = Field(..., pattern=r"^\d+/(second|minute|hour|day)$")
    batch_create: str = Field(..., pattern=r"^\d+/(second|minute|hour|day)$")


@router.get(
    "/rate-limits",
    response_model=RateLimitsResponse,
    summary="Get current rate limits (admin only)",
)
async def get_rate_limits(
    admin: User = Depends(require_admin),
):
    return RateLimitsResponse(
        agent_chat=get_rate_limit("agent_chat"),
        batch_create=get_rate_limit("batch_create"),
    )


@router.put(
    "/rate-limits",
    response_model=RateLimitsResponse,
    summary="Update rate limits (admin only)",
)
async def update_rate_limits(
    body: RateLimitsUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    limits = {"agent_chat": body.agent_chat, "batch_create": body.batch_create}

    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "rate_limits")
    )
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = json.dumps(limits)
        db.add(setting)
    else:
        db.add(SystemSetting(key="rate_limits", value=json.dumps(limits)))
    await db.flush()

    update_rate_limit_cache(limits)

    logger.info("Rate limits updated by %s: %s", admin.username, limits)
    return RateLimitsResponse(**limits)
