"""Base parser interface for IDP Kit document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParseResult:
    """Result of parsing a document.

    Attributes
    ----------
    text:
        The full extracted text content of the document.
    pages:
        A list of per-page dicts, each containing at least ``"page"`` (int)
        and ``"text"`` (str).
    metadata:
        Arbitrary metadata extracted from the document (title, author, etc.).
    page_count:
        Total number of pages (or logical sections) in the document.
    """

    text: str
    pages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    page_count: int = 0


class BaseParser(ABC):
    """Abstract base class for document parsers.

    Every parser must implement :meth:`parse` (which extracts text and metadata
    from a file) and :meth:`supported_extensions` (which declares the file
    extensions the parser can handle).
    """

    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """Parse a document file and return extracted content.

        Parameters
        ----------
        file_path:
            Absolute path to the document file on disk.

        Returns
        -------
        ParseResult
            The extracted text, per-page breakdown, metadata, and page count.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return the file extensions this parser supports.

        Each extension should be lowercase without the leading dot,
        e.g. ``["pdf"]``.
        """
        ...
