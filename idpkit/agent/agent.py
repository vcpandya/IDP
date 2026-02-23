"""IDP Kit Agent — AI orchestrator with LLM function-calling loop.

The IDPAgent receives user messages, decides which tools to call,
executes them, feeds results back to the LLM, and loops until the
model produces a final text response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import litellm
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from idpkit.agent.memory import ConversationMemory
from idpkit.agent.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# Maximum iterations of the tool-calling loop to prevent runaway chains.
MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are an intelligent document processing assistant powered by IDP Kit.

You help users analyze, search, summarize, and extract information from their
uploaded documents.  You have access to the following tools:

- **search_document**: Search a document's tree index for sections relevant
  to a query.
- **list_documents**: List all documents the user has access to.
- **summarize_section**: Summarize a specific section (node) of a document.
- **extract_data**: Extract structured data (tables, entities, key facts,
  financial data, dates, etc.) from a document.
- **query_graph**: Query the knowledge graph for entity information.
  Operations: find_entity (search by name), entity_mentions (where an entity
  appears), related_sections (sections sharing entities), cross_document_links
  (linked documents), document_entities (all entities in a document).
- **find_cross_references**: Find all sections across all documents that
  mention a given topic or entity. Use this to answer "Where is X mentioned?"
  or "Which documents discuss Y?"

Guidelines:
1. Always be helpful, accurate, and concise.
2. When the user asks about a specific document, use search_document first to
   find relevant sections, then answer based on the results.
3. If you need to know which documents are available, call list_documents.
4. Cite the document filename and section titles when referencing information.
5. If a tool returns an error, explain it to the user in plain language.
6. If no documents are loaded, suggest the user upload one first.
7. When asked where something is mentioned or which documents discuss a topic,
   use find_cross_references or query_graph to leverage the knowledge graph.
8. When asked about relationships between concepts, use query_graph to explore
   the entity graph and find connections.
"""


class IDPAgent:
    """Orchestrates multi-turn conversations with LLM tool-calling.

    Usage::

        agent = IDPAgent()
        result = await agent.chat(
            message="Summarize section 3",
            document_ids=["abc-123"],
            llm=llm_client,
            db=db_session,
        )
        print(result["response"])
    """

    def __init__(self) -> None:
        self._system_prompt = SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        document_ids: list[str],
        llm: LLMClient,
        db: AsyncSession,
        conversation: Optional[ConversationMemory] = None,
    ) -> dict:
        """Process a user message through the tool-calling loop.

        Args:
            message: The user's natural-language message.
            document_ids: IDs of documents in scope for this conversation.
            llm: An ``LLMClient`` instance for LLM calls.
            db: An async SQLAlchemy session for database access.
            conversation: Optional ``ConversationMemory`` to maintain context
                across turns.  A fresh one is created if not supplied.

        Returns:
            A dict with:
            - ``response``: The final text response from the assistant.
            - ``tool_calls``: A list of ``{"name", "args", "result"}`` dicts
              for every tool call made during this turn.
        """
        if conversation is None:
            conversation = ConversationMemory()

        # Record the user message
        conversation.add_message("user", message)

        # Build the messages list for the LLM
        messages = self._build_messages(conversation, document_ids)

        tool_call_log: list[dict] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            # Call the LLM with tool definitions
            try:
                response = await litellm.acompletion(
                    model=llm.default_model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=llm.temperature,
                    api_key=llm.api_key or None,
                    api_base=llm.api_base or None,
                )
            except Exception as exc:
                logger.error("Agent LLM call failed (iteration %d): %s", iteration, exc)
                error_msg = f"I encountered an error communicating with the language model: {exc}"
                conversation.add_message("assistant", error_msg)
                return {"response": error_msg, "tool_calls": tool_call_log}

            choice = response.choices[0]
            assistant_message = choice.message

            # If the model wants to call tools
            if assistant_message.tool_calls:
                # Append the assistant message (with tool_calls) to history
                messages.append(assistant_message.model_dump())

                for tool_call in assistant_message.tool_calls:
                    fn = tool_call.function
                    tool_name = fn.name
                    try:
                        tool_args = json.loads(fn.arguments) if fn.arguments else {}
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(
                        "Agent calling tool '%s' with args: %s",
                        tool_name,
                        json.dumps(tool_args, default=str)[:200],
                    )

                    # Execute the tool
                    try:
                        tool_result = await execute_tool(
                            name=tool_name,
                            args=tool_args,
                            llm=llm,
                            db=db,
                        )
                    except Exception as exc:
                        logger.error("Tool '%s' execution failed: %s", tool_name, exc)
                        tool_result = {"error": f"Tool execution failed: {exc}"}

                    tool_call_log.append({
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    })

                    # Record the tool result in conversation memory
                    result_str = json.dumps(tool_result, default=str)
                    conversation.add_message(
                        "tool",
                        result_str,
                        tool_name=tool_name,
                        tool_result=tool_result,
                    )

                    # Append tool response to messages for next LLM call
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })

                # Continue the loop so the LLM can process tool results
                continue

            # No tool calls — the model produced a final text response
            final_text = assistant_message.content or ""
            conversation.add_message("assistant", final_text)

            return {"response": final_text, "tool_calls": tool_call_log}

        # Exhausted iterations — return whatever we have
        fallback = "I've reached the maximum number of reasoning steps. Here's what I found so far based on the tool results."
        conversation.add_message("assistant", fallback)
        return {"response": fallback, "tool_calls": tool_call_log}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        conversation: ConversationMemory,
        document_ids: list[str],
    ) -> list[dict]:
        """Build the full messages list including system prompt and history."""
        doc_context = ""
        if document_ids:
            doc_context = (
                f"\n\nThe user has the following document IDs in scope: "
                f"{', '.join(document_ids)}. You can use these IDs when calling tools."
            )

        system_msg = {
            "role": "system",
            "content": self._system_prompt + doc_context,
        }

        # Get conversation history (only role + content for LLM compatibility)
        history = conversation.get_messages(limit=20)

        return [system_msg] + history
