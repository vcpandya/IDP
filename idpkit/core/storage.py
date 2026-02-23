"""IDP Kit storage backend interface and implementations."""

import io
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Optional

import httpx

from .exceptions import StorageError

logger = logging.getLogger(__name__)

REPLIT_SIDECAR_ENDPOINT = "http://127.0.0.1:1106"


class StorageBackend(ABC):
    """Abstract interface for file storage operations."""

    @abstractmethod
    def save(self, key: str, data: bytes | BinaryIO) -> str:
        """Save data and return the storage path/key."""
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data by key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data by key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        ...

    @abstractmethod
    def get_path(self, key: str) -> Optional[str]:
        """Get the filesystem path for a key, if applicable."""
        ...


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        resolved = (self.base_path / key).resolve()
        if not str(resolved).startswith(str(self.base_path.resolve())):
            raise StorageError(f"Path traversal detected: {key}")
        return resolved

    def save(self, key: str, data: bytes | BinaryIO) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            with open(path, "wb") as f:
                shutil.copyfileobj(data, f)
        return str(path)

    def load(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"File not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        search_path = self._resolve(prefix) if prefix else self.base_path
        if not search_path.exists():
            return []
        keys = []
        for p in search_path.rglob("*"):
            if p.is_file():
                keys.append(str(p.relative_to(self.base_path)))
        return sorted(keys)

    def get_path(self, key: str) -> Optional[str]:
        path = self._resolve(key)
        return str(path) if path.exists() else None


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend using Replit's object storage sidecar."""

    def __init__(self, bucket_id: str, private_dir: str):
        self.bucket_id = bucket_id
        self.private_dir = private_dir.rstrip("/")
        self._cache_dir = Path(tempfile.mkdtemp(prefix="idpkit_gcs_"))

    def _object_name(self, key: str) -> str:
        return f"{self.private_dir}/{key}"

    def _sign_url(self, object_name: str, method: str, ttl_sec: int = 900) -> str:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_sec)).isoformat()
        payload = {
            "bucket_name": self.bucket_id,
            "object_name": object_name,
            "method": method,
            "expires_at": expires_at,
        }
        resp = httpx.post(
            f"{REPLIT_SIDECAR_ENDPOINT}/object-storage/signed-object-url",
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            raise StorageError(
                f"Failed to sign URL ({method} {object_name}): "
                f"status {resp.status_code}, body: {resp.text[:200]}"
            )
        return resp.json()["signed_url"]

    def save(self, key: str, data: bytes | BinaryIO) -> str:
        obj_name = self._object_name(key)
        upload_url = self._sign_url(obj_name, "PUT")

        if isinstance(data, (bytes, bytearray)):
            content = data
        else:
            content = data.read()

        resp = httpx.put(
            upload_url,
            content=content,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )
        if resp.status_code not in (200, 201):
            raise StorageError(
                f"Failed to upload {key}: status {resp.status_code}"
            )

        cache_path = self._cache_dir / key
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content if isinstance(content, bytes) else content)

        return key

    def load(self, key: str) -> bytes:
        cache_path = self._cache_dir / key
        if cache_path.exists():
            return cache_path.read_bytes()

        obj_name = self._object_name(key)
        download_url = self._sign_url(obj_name, "GET")
        resp = httpx.get(download_url, timeout=120)
        if resp.status_code == 404:
            raise StorageError(f"File not found: {key}")
        if resp.status_code != 200:
            raise StorageError(
                f"Failed to download {key}: status {resp.status_code}"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)

        return resp.content

    def delete(self, key: str) -> None:
        obj_name = self._object_name(key)
        try:
            delete_url = self._sign_url(obj_name, "DELETE")
            httpx.delete(delete_url, timeout=30)
        except Exception as exc:
            logger.warning("Failed to delete %s from GCS: %s", key, exc)

        cache_path = self._cache_dir / key
        if cache_path.exists():
            if cache_path.is_file():
                cache_path.unlink()
            elif cache_path.is_dir():
                shutil.rmtree(cache_path)

    def exists(self, key: str) -> bool:
        cache_path = self._cache_dir / key
        if cache_path.exists():
            return True

        obj_name = self._object_name(key)
        try:
            head_url = self._sign_url(obj_name, "HEAD")
            resp = httpx.head(head_url, timeout=15)
            return resp.status_code == 200
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        cache_search = self._cache_dir / prefix if prefix else self._cache_dir
        if not cache_search.exists():
            return []
        keys = []
        for p in cache_search.rglob("*"):
            if p.is_file():
                keys.append(str(p.relative_to(self._cache_dir)))
        return sorted(keys)

    def get_path(self, key: str) -> Optional[str]:
        cache_path = self._cache_dir / key
        if cache_path.exists():
            return str(cache_path)

        try:
            data = self.load(key)
            return str(cache_path)
        except StorageError:
            return None
