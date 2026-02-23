"""IDP Kit Auth API routes — registration, login, logout, API keys."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db
from idpkit.db.models import User
from idpkit.api.deps import (
    get_current_user,
    hash_password,
    verify_password,
    create_access_token,
    generate_api_key,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    email: Optional[str] = Field(None, max_length=255)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: int
    api_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyResponse(BaseModel):
    api_key: str


class MessageResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user account.

    Returns the created user on success.  Raises 409 if the username or email
    is already taken.
    """
    # Check uniqueness — username
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    # Check uniqueness — email (if provided)
    if body.email:
        result = await db.execute(select(User).where(User.email == body.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        email=body.email,
        is_active=0,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    logger.info("New user registered (pending approval): %s (id=%s)", user.username, user.id)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and obtain a JWT token",
)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate with username/password.

    Returns a JWT bearer token in the response body **and** sets an
    ``session_token`` HTTP-only cookie so browsers can use cookie-based auth.
    """
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is pending admin approval. Please contact an administrator.",
        )

    token = create_access_token(user.id)

    # Set session cookie for browser clients
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    logger.info("User logged in: %s", user.username)
    return TokenResponse(access_token=token)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Log out (clear session cookie)",
)
async def logout(
    response: Response,
    _user: User = Depends(get_current_user),
):
    """Clear the session cookie.

    The JWT itself cannot be revoked; this only removes the browser cookie.
    Clients using the Bearer header should discard the token on their side.
    """
    response.delete_cookie(
        key="session_token",
        httponly=True,
        samesite="lax",
    )
    return MessageResponse(detail="Logged out successfully")


@router.post(
    "/apikey",
    response_model=ApiKeyResponse,
    summary="Generate a new API key for the current user",
)
async def create_api_key(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate (or regenerate) an API key.

    The previous API key (if any) is replaced.  The raw key is returned
    **once** — it cannot be retrieved later.
    """
    new_key = generate_api_key()
    user.api_key = new_key
    db.add(user)
    await db.flush()

    logger.info("API key generated for user %s", user.username)
    return ApiKeyResponse(api_key=new_key)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user info",
)
async def get_me(
    user: User = Depends(get_current_user),
):
    """Return the authenticated user's profile information."""
    return user
