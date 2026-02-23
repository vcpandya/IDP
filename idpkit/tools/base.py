"""Base class for all IDP Kit Smart Tools."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolOptions, ToolResult


class BaseTool(ABC):
    """Abstract base class for Smart Tools.

    All Smart Tools inherit from this class and implement the execute() method.
    Tools are auto-discovered and registered as:
    - API endpoints: POST /api/tools/{name}
    - Agent function-calling tools
    - UI cards in the Tools panel
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g., 'smart_summary')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Smart Summary')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the tool does."""
        ...

    @property
    def options_schema(self) -> dict:
        """JSON Schema for tool-specific options. Override in subclass."""
        return {}

    @abstractmethod
    async def execute(
        self,
        document_id: str,
        options: dict,
        llm: LLMClient,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the tool on a document.

        Args:
            document_id: ID of the document to process.
            options: Tool-specific options matching options_schema.
            llm: LLMClient instance for AI operations.
            **kwargs: Additional context (db session, storage, etc.)

        Returns:
            ToolResult with status, data, and optional output file.
        """
        ...

    def validate_options(self, options: dict) -> dict:
        """Validate and normalize options. Override for custom validation."""
        return options

    def to_agent_tool(self) -> dict:
        """Convert to an agent function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.options_schema or {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Document ID to process"},
                    },
                    "required": ["document_id"],
                },
            },
        }
