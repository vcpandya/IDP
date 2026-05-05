"""IDP Kit storage backend interface and implementations."""

import io
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import httpx

from .exceptions import StorageError

logger = logging.getLogger(__name__)

REPLIT_SIDECAR_ENDPOINT = "http://127.0.0.1:1106"


class StorageBackend(ABC):
    """Abstract interface for file storage operations."""

    @property
    def supports_signed_urls(self) -> bool:
        return False

    def get_signed_upload_url(self, key: str, content_type: str = "application/octet-stream", ttl_sec: int = 900) -> Optional[str]:
        return None

    @abstractmethod
    def save(self, key: str, data: bytes | BinaryIO) -> str:
        """Save data and return the storage path/key."""
        ...

    async def put(self, key: str, data: bytes | BinaryIO) -> str:
        """Async alias for save()."""
        return self.save(key, data)

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data by key."""
        ...

    def iter_bytes(self, key: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        """Yield the object's bytes in chunks without buffering the entire file.

        Default implementation falls back to ``load(key)`` so subclasses that
        cannot stream remain functional. Override for true streaming.
        """
        data = self.load(key)
        for offset in range(0, len(data), chunk_size):
            yield data[offset:offset + chunk_size]

    def peek_bytes(self, key: str, n: int = 512) -> bytes:
        """Return up to ``n`` leading bytes without poisoning any download
        cache. The default implementation uses ``load(key)`` and slices, which
        is safe for in-memory backends; backends that lazily cache to disk
        (e.g. GCS) MUST override this so a partial fetch does not leave a
        truncated cache file behind that subsequent reads would serve."""
        return self.load(key)[:n]

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

    def iter_bytes(self, key: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"File not found: {key}")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def peek_bytes(self, key: str, n: int = 512) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"File not found: {key}")
        with open(path, "rb") as f:
            return f.read(n)

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

    @property
    def supports_signed_urls(self) -> bool:
        return True

    def get_signed_upload_url(self, key: str, content_type: str = "application/octet-stream", ttl_sec: int = 900) -> Optional[str]:
        obj_name = self._object_name(key)
        return self._sign_url(obj_name, "PUT", ttl_sec)

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

    def iter_bytes(self, key: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        cache_path = self._cache_dir / key
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return

        obj_name = self._object_name(key)
        download_url = self._sign_url(obj_name, "GET")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Stream into a sibling .tmp file and atomically rename only after
        # the HTTP body completes — otherwise an interrupted iter_bytes()
        # (caller breaks early, network error, etc.) would leave a truncated
        # cache file that subsequent reads would happily serve as the full
        # object.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        ok = False
        try:
            with httpx.stream("GET", download_url, timeout=120) as resp:
                if resp.status_code == 404:
                    raise StorageError(f"File not found: {key}")
                if resp.status_code != 200:
                    raise StorageError(
                        f"Failed to download {key}: status {resp.status_code}"
                    )
                with open(tmp_path, "wb") as cache_f:
                    for chunk in resp.iter_bytes(chunk_size):
                        cache_f.write(chunk)
                        yield chunk
            os.replace(tmp_path, cache_path)
            ok = True
        finally:
            if not ok and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:  # pragma: no cover
                    pass

    def peek_bytes(self, key: str, n: int = 512) -> bytes:
        # If we already have the full object cached, slice it locally.
        cache_path = self._cache_dir / key
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return f.read(n)
        # Otherwise issue a Range request so we never write a truncated
        # cache file. The cache is left untouched; a subsequent download
        # will fetch the full object.
        obj_name = self._object_name(key)
        download_url = self._sign_url(obj_name, "GET")
        headers = {"Range": f"bytes=0-{max(0, n - 1)}"}
        resp = httpx.get(download_url, headers=headers, timeout=30)
        if resp.status_code == 404:
            raise StorageError(f"File not found: {key}")
        if resp.status_code not in (200, 206):
            raise StorageError(
                f"Failed to peek {key}: status {resp.status_code}"
            )
        return resp.content[:n]

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
