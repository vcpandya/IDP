"""Fernet-based credential encryption.

The Fernet key is derived deterministically from the application's
SECRET_KEY (sha256 → urlsafe base64). This means:
- Re-deploying with the same SECRET_KEY preserves access to stored creds.
- Rotating SECRET_KEY invalidates all stored connections (users must re-auth).
- The plaintext SECRET_KEY itself is never stored; only the Fernet key it derives.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    # Lazy import to ensure SECRET_KEY env resolution after deps.py initialises.
    from idpkit.api.deps import SECRET_KEY

    digest = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_credentials(payload: dict[str, Any]) -> str:
    """Encrypt a credential mapping; returns a urlsafe-base64 string."""
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _fernet().encrypt(raw).decode("ascii")


def decrypt_credentials(token: str) -> dict[str, Any]:
    """Decrypt a credential token. Raises ValueError on tamper / wrong key."""
    try:
        raw = _fernet().decrypt(token.encode("ascii"))
    except InvalidToken as exc:
        raise ValueError("Stored credentials are unreadable (key changed or tampered)") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Decrypted credentials are not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Decrypted credentials are not a mapping")
    return data
