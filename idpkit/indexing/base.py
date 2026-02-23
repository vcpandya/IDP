"""Base indexer interface for IDP Kit."""

from abc import ABC, abstractmethod
from typing import Any


class BaseIndexer(ABC):
    """Abstract base class for document indexers.

    Every indexer must implement ``build_index`` (which produces the
    tree-structured index dict) and ``supported_formats`` (which declares
    which file extensions the indexer can handle).
    """

    @abstractmethod
    async def build_index(self, source: Any, **options) -> dict:
        """Build a tree index from *source*.

        Parameters
        ----------
        source:
            A file path (``str``) or an in-memory object (e.g. ``BytesIO``)
            that the concrete indexer knows how to process.
        **options:
            Arbitrary keyword arguments forwarded to the underlying engine
            (model name, feature flags, thresholds, etc.).

        Returns
        -------
        dict
            Must contain at least ``doc_name`` (str) and ``structure`` (list).
            May optionally contain ``doc_description`` (str).
        """
        ...

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return the file extensions this indexer supports.

        Each extension must include the leading dot, e.g. ``[".pdf"]``.
        """
        ...
