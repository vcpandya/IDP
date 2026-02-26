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

litellm.drop_params = True

from .exceptions import LLMError, LLMMaxRetriesError
from .schemas import LLMResponse

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


def _resolve_api_key_for_model(model: str) -> Optional[str]:
    """Return the correct API key for a model based on its provider prefix.

    LiteLLM expects specific env var names per provider, but users may have
    differently-named keys (e.g. GOOGLE_API_KEY instead of GEMINI_API_KEY).
    This function bridges that gap.
    """
    m = model.lower()

    if m.startswith("gemini/") or m.startswith("google/"):
        return (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

    if m.startswith("openrouter/"):
        return os.getenv("OPENROUTER_API_KEY")

    if any(m.startswith(p) for p in ("claude-", "anthropic/")):
        return os.getenv("ANTHROPIC_API_KEY")

    if any(m.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-", "openai/", "ft:gpt-")):
        return os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")

    return None


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

    default_model: str = "gpt-4o-mini"
    temperature: float = 0
    max_retries: int = 10
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def _get_completion_kwargs(
        self,
        prompt,
        model: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """Build kwargs dict for litellm.completion/acompletion.

        ``prompt`` may be a plain string **or** a list of content blocks
        for multimodal messages (e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``).
        """
        model = model or self.default_model

        content = prompt if isinstance(prompt, list) else prompt

        if chat_history:
            messages = list(chat_history)
            messages.append({"role": "user", "content": content})
        else:
            messages = [{"role": "user", "content": content}]

        completion_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        else:
            resolved_key = _resolve_api_key_for_model(model)
            if resolved_key:
                completion_kwargs["api_key"] = resolved_key
                logger.debug("Resolved API key for model %s: %s...%s (len=%d)",
                             model, resolved_key[:4], resolved_key[-4:], len(resolved_key))
            else:
                logger.debug("No API key resolved for model %s, relying on LiteLLM env defaults", model)
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
            except litellm.AuthenticationError as e:
                logger.error(f"Authentication error for model {used_model}: {e}")
                raise LLMMaxRetriesError(
                    f"Authentication failed for {used_model}: {e}"
                ) from e
            except litellm.UnsupportedParamsError as e:
                logger.error(f"Unsupported params for model {used_model}: {e}")
                raise LLMMaxRetriesError(
                    f"Unsupported parameters for {used_model}: {e}"
                ) from e
            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached for model {used_model}")
                    raise LLMMaxRetriesError(
                        f"Max retries ({self.max_retries}) reached: {e}"
                    ) from e

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
            except litellm.AuthenticationError as e:
                logger.error(f"Authentication error for model {used_model}: {e}")
                raise LLMMaxRetriesError(
                    f"Authentication failed for {used_model}: {e}"
                ) from e
            except litellm.UnsupportedParamsError as e:
                logger.error(f"Unsupported params for model {used_model}: {e}")
                raise LLMMaxRetriesError(
                    f"Unsupported parameters for {used_model}: {e}"
                ) from e
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
        resolved_model = model or os.getenv("IDP_DEFAULT_MODEL", "gpt-4o-mini")

        client = LLMClient(
            default_model=resolved_model,
            api_key=api_key,
        )

        if not api_key and not model:
            _default_client = client

        return client

    return _default_client


def reset_default_client() -> None:
    """Reset the default client (useful for testing)."""
    global _default_client
    _default_client = None
