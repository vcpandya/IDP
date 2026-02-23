"""IDP Kit Agent conversation memory management."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """In-memory conversation history manager.

    Stores messages as a list of dicts compatible with the LiteLLM/OpenAI
    chat message format.  Each message has at minimum ``role`` and ``content``.
    Tool-related messages carry additional ``tool_name`` and ``tool_result``
    fields for traceability.
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_result: object = None,
    ) -> None:
        """Append a message to the conversation history.

        Args:
            role: One of ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``.
            content: The text content of the message.
            tool_name: If the message is from a tool, the tool's name.
            tool_result: If the message is from a tool, the structured result.
        """
        msg: dict = {"role": role, "content": content}
        if tool_name is not None:
            msg["tool_name"] = tool_name
        if tool_result is not None:
            msg["tool_result"] = tool_result
        self._messages.append(msg)

    def get_messages(self, limit: int = 20) -> list[dict]:
        """Return the most recent *limit* messages.

        Only the ``role`` and ``content`` keys are included so the list can
        be passed directly to the LLM as ``chat_history``.
        """
        recent = self._messages[-limit:] if limit else self._messages
        # Return only LLM-compatible keys
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_full_messages(self, limit: int = 20) -> list[dict]:
        """Return the most recent messages including tool metadata."""
        return list(self._messages[-limit:]) if limit else list(self._messages)

    def clear(self) -> None:
        """Remove all messages from the conversation history."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self._messages)})"
