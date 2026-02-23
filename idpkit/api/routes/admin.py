"""IDP Kit Admin API routes — user management and approval."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User
from idpkit.api.deps import require_admin

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


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all users (admin only)",
)
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all registered users with their status."""
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
    """Activate a pending user account."""
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
    """Deactivate an active user account."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

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
    """Permanently delete a user account and all their data."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    username = user.username
    await db.delete(user)
    await db.flush()

    logger.info("User deleted by %s: %s (id=%s)", admin.username, username, user_id)
    return MessageResponse(detail=f"User '{username}' has been deleted")
