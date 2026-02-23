"""FastAPI dependency injection — auth, database, storage."""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import json

from slowapi import Limiter
from slowapi.util import get_remote_address

from idpkit.db.session import get_db
from idpkit.db.models import User, SystemSetting, UserRole
from idpkit.core.storage import LocalStorageBackend, GCSStorageBackend, StorageBackend
from idpkit.core.llm import LLMClient, get_default_client

DEFAULT_RATE_LIMITS = {
    "agent_chat": "30/minute",
    "batch_create": "10/minute",
}

_rate_limit_cache: dict[str, str] = {}


def get_rate_limit(key: str) -> str:
    if key in _rate_limit_cache:
        return _rate_limit_cache[key]
    return DEFAULT_RATE_LIMITS.get(key, "60/minute")


def update_rate_limit_cache(limits: dict[str, str]):
    _rate_limit_cache.update(limits)


def _admin_exempt_key_func(request: Request) -> str:
    user_role = getattr(request.state, "_user_role", None)
    if user_role in (UserRole.ADMIN.value, UserRole.SUPERADMIN.value):
        return "__admin_exempt__"
    return get_remote_address(request)


limiter = Limiter(
    key_func=_admin_exempt_key_func,
    default_limits=[],
)

import logging as _logging

_deps_logger = _logging.getLogger(__name__)

_secret = os.getenv("SESSION_SECRET") or os.getenv("IDP_SECRET_KEY")
if not _secret:
    _secret = secrets.token_urlsafe(64)
    _deps_logger.warning(
        "No SESSION_SECRET env var set — using an ephemeral random key. "
        "Sessions will not survive server restarts."
    )
SECRET_KEY = _secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# --- Password hashing ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# --- JWT tokens ---

def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# --- API key generation ---

def generate_api_key() -> str:
    return f"idpk_{secrets.token_urlsafe(32)}"


# --- FastAPI dependencies ---

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate via JWT Bearer token, API key header, or session cookie."""
    user_id = None

    # 1. Try Bearer token (JWT)
    if credentials and credentials.credentials:
        user_id = decode_token(credentials.credentials)

    # 2. Try API key header
    if not user_id:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            result = await db.execute(select(User).where(User.api_key == api_key))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return user

    # 3. Try session cookie
    if not user_id:
        session_token = request.cookies.get("session_token")
        if session_token:
            user_id = decode_token(session_token)

    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Like get_current_user but returns None instead of raising."""
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None


_storage_instance: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    bucket_id = os.getenv("DEFAULT_OBJECT_STORAGE_BUCKET_ID")
    private_dir = os.getenv("PRIVATE_OBJECT_DIR")
    if bucket_id and private_dir:
        _storage_instance = GCSStorageBackend(bucket_id=bucket_id, private_dir=private_dir)
    else:
        path = os.getenv("IDP_STORAGE_PATH", "./storage")
        _storage_instance = LocalStorageBackend(path)
    return _storage_instance


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Require the current user to have admin or superadmin role."""
    if user.role not in (UserRole.ADMIN.value, UserRole.SUPERADMIN.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_superadmin(
    user: User = Depends(get_current_user),
) -> User:
    """Require the current user to have superadmin role."""
    if user.role != UserRole.SUPERADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required",
        )
    return user


def get_llm() -> LLMClient:
    return get_default_client()
