"""
IDP Kit LLM Client — Unified multi-provider LLM interface.

Wraps LiteLLM to provide a consistent API for all LLM providers:
OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter, and 100+ more.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import litellm

from .exceptions import LLMError, LLMMaxRetriesError
from .schemas import LLMResponse

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class LLMClient:
    """Unified LLM client wrapping LiteLLM for multi-provider support.

    Supports OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter,
    and all other LiteLLM-supported providers.

    Usage:
        client = LLMClient(default_model="gpt-4o")
        response = client.complete("What is 2+2?")
        print(response.content)

        # Async
        response = await client.acomplete("What is 2+2?")

        # With chat history
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        response = client.complete("How are you?", chat_history=history)

        # Different provider
        client = LLMClient(default_model="claude-sonnet-4-20250514")
        response = client.complete("Explain quantum computing")
    """

    default_model: str = "gpt-4o-2024-11-20"
    temperature: float = 0
    max_retries: int = 10
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def _get_completion_kwargs(
        self,
        prompt: str,
        model: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """Build kwargs dict for litellm.completion/acompletion."""
        model = model or self.default_model

        if chat_history:
            messages = list(chat_history)
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]

        completion_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Pass API key if explicitly set
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        # Forward any extra kwargs (tools, response_format, etc.)
        for key in ("tools", "tool_choice", "response_format", "max_tokens", "top_p", "stop"):
            if key in kwargs:
                completion_kwargs[key] = kwargs[key]

        return completion_kwargs

    def _parse_response(self, response, model: str) -> LLMResponse:
        """Parse a LiteLLM response into our LLMResponse model."""
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            model=model,
            usage=usage,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous LLM completion with retry logic.

        Args:
            prompt: The user message to send.
            model: Override the default model.
            chat_history: Previous conversation messages.
            **kwargs: Extra params (temperature, tools, max_tokens, etc.)

        Returns:
            LLMResponse with content, finish_reason, model, usage.

        Raises:
            LLMMaxRetriesError: If all retries are exhausted.
        """
        completion_kwargs = self._get_completion_kwargs(
            prompt, model, chat_history, **kwargs
        )
        used_model = completion_kwargs["model"]

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(**completion_kwargs)
                return self._parse_response(response, used_model)
            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached for model {used_model}")
                    raise LLMMaxRetriesError(
                        f"Max retries ({self.max_retries}) reached: {e}"
                    ) from e

        # Should not reach here, but just in case
        raise LLMMaxRetriesError("Max retries reached")

    async def acomplete(
        self,
        prompt: str,
        model: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs,
    ) -> LLMResponse:
        """Async LLM completion with retry logic.

        Same interface as complete() but async.
        """
        completion_kwargs = self._get_completion_kwargs(
            prompt, model, chat_history, **kwargs
        )
        used_model = completion_kwargs["model"]

        for attempt in range(self.max_retries):
            try:
                response = await litellm.acompletion(**completion_kwargs)
                return self._parse_response(response, used_model)
            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached for model {used_model}")
                    raise LLMMaxRetriesError(
                        f"Max retries ({self.max_retries}) reached: {e}"
                    ) from e

        raise LLMMaxRetriesError("Max retries reached")

    def complete_with_finish_reason(
        self,
        prompt: str,
        model: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs,
    ) -> tuple[str, str]:
        """Completion that returns (content, finish_reason) tuple.

        Backward compatible with the engine's ChatGPT_API_with_finish_reason.
        finish_reason is "finished" or "max_output_reached".
        """
        response = self.complete(prompt, model, chat_history, **kwargs)
        if response.finish_reason == "length":
            return response.content, "max_output_reached"
        return response.content, "finished"


# Global default client instance
_default_client: Optional[LLMClient] = None


def get_default_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Get or create the default LLMClient instance.

    Uses environment variables for configuration:
    - CHATGPT_API_KEY or OPENAI_API_KEY: OpenAI API key
    - IDP_DEFAULT_MODEL: Default model to use
    - ANTHROPIC_API_KEY: For Anthropic models
    - GOOGLE_API_KEY: For Google Gemini models
    - OPENROUTER_API_KEY: For OpenRouter
    """
    global _default_client

    if _default_client is None or api_key or model:
        resolved_key = api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_model = model or os.getenv("IDP_DEFAULT_MODEL", "gpt-4o-2024-11-20")

        # Set API keys for LiteLLM to discover
        if resolved_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = resolved_key

        client = LLMClient(
            default_model=resolved_model,
            api_key=resolved_key,
        )

        if not api_key and not model:
            _default_client = client

        return client

    return _default_client


def reset_default_client() -> None:
    """Reset the default client (useful for testing)."""
    global _default_client
    _default_client = None
