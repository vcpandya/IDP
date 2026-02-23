"""IDP Kit storage backend interface and implementations."""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional

from .exceptions import StorageError


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
