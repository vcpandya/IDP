"""IDP Kit event/hook system for extensibility."""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Event types
DOCUMENT_UPLOADED = "document.uploaded"
DOCUMENT_INDEXED = "document.indexed"
DOCUMENT_DELETED = "document.deleted"
QUERY_RECEIVED = "query.received"
QUERY_COMPLETED = "query.completed"
TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
JOB_STARTED = "job.started"
JOB_COMPLETED = "job.completed"
JOB_FAILED = "job.failed"


class EventBus:
    """Simple event bus for plugin hooks and internal events."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, handler: Callable) -> None:
        """Register a handler for an event."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister a handler for an event."""
        self._handlers[event] = [
            h for h in self._handlers[event] if h is not handler
        ]

    async def emit(self, event: str, **kwargs: Any) -> list[Any]:
        """Emit an event, calling all registered handlers."""
        results = []
        for handler in self._handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**kwargs)
                else:
                    result = handler(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")
        return results

    def emit_sync(self, event: str, **kwargs: Any) -> list[Any]:
        """Emit an event synchronously (for non-async contexts)."""
        results = []
        for handler in self._handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    logger.warning(
                        f"Async handler for {event} called in sync context, skipping"
                    )
                    continue
                result = handler(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")
        return results


# Global event bus instance
event_bus = EventBus()
