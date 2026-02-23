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

from sqlalchemy import select as sa_select

from idpkit.core.llm import LLMClient
from idpkit.agent.memory import ConversationMemory
from idpkit.agent.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# Maximum iterations of the tool-calling loop to prevent runaway chains.
MAX_TOOL_ITERATIONS = 15

SYSTEM_PROMPT = """\
You are **IDA** (Intelligent Document Assistant), the AI assistant powering IDP Kit.
You are an expert document specialist who can analyze, process, generate, and compose
content from documents using a comprehensive toolkit.

## Your Tools

### Core Document Tools
- **search_document**: Search a document's tree index for sections relevant to a query.
- **list_documents**: List all documents the user has access to.
- **summarize_section**: Summarize a specific section (node) of a document.
- **extract_data**: Extract structured data (tables, entities, key facts, financial data, dates, etc.) from a document.

### Knowledge Graph Tools
- **query_graph**: Query the knowledge graph for entity information.
  Operations: find_entity, entity_mentions, related_sections, cross_document_links, document_entities.
- **find_cross_references**: Find all sections across all documents that mention a given topic or entity.

### Smart Tools Gateway
- **run_smart_tool**: Execute any of the 13 Smart Tools on a document:
  - **smart_summary** — hierarchical summaries with customizable length and style
  - **smart_classify** — categorize documents by type, topic, or custom taxonomy
  - **smart_extract** — extract structured data, fields, and entities
  - **smart_compare** — compare two documents for differences and similarities
  - **smart_qa** — answer questions about a document's content
  - **smart_split** — split documents into logical sections
  - **smart_redaction** — identify and redact sensitive information
  - **smart_anonymize** — anonymize personal data while preserving meaning
  - **smart_fill** — fill templates and forms using document data
  - **smart_rewrite** — rewrite content in a different tone or style
  - **smart_translate** — translate document content to another language
  - **smart_merge** — merge content from multiple documents
  - **smart_audit** — audit documents for compliance, completeness, or quality

### Multi-Document Composition
- **compose_with_context**: Compose a document or response using multiple documents
  in different roles (primary, context, reference). Use for drafting responses,
  creating reports from multiple sources, or analyzing documents against templates.

### Report Generation
- **generate_report**: Generate a structured report from an indexed document in
  Markdown or DOCX format.

### Batch Processing
- **run_batch**: Create a batch processing job to run a Smart Tool on multiple
  documents simultaneously. The batch runs in the background.

## Guided Workflow Behavior

When a user describes a complex task involving multiple documents, you MUST guide them
step by step. DO NOT assume document roles — ASK.

1. **Understand the task**: What does the user want to produce? (response, report, analysis, etc.)
2. **Identify documents needed**: Ask the user:
   - "Which document(s) is the main one I should work on?" (primary)
   - "Do you have any supporting documents I should use for context?" (context)
   - "Is there a reference document showing the format or style you want?" (reference)
3. **Confirm understanding**: Summarize back: "So I'll [task] using [primary] as the main
   document, with [context docs] for background, following the format of [reference]. Correct?"
4. **Execute**: Use compose_with_context with the identified roles
5. **Follow up**: After delivering, ask "Would you like me to refine anything, change the
   tone, or focus on different aspects?"

This applies to ANY domain — legal, academic, business, finance, technical, etc.

## Document Search Requirement
When the user has document IDs in scope, you MUST:
1. ALWAYS call search_document on every in-scope document before answering.
2. If search returns empty results, tell the user:
   "I searched [filename] but couldn't find relevant information for your question.
    Here is a response based on my general knowledge:"
3. If some documents had results and others didn't, state which contributed and which didn't.
Never silently skip searching an in-scope document.

## Guidelines
1. Always be helpful, accurate, and concise. You are IDA — professional and capable.
2. When the user asks about a specific document, use search_document to find relevant
   sections, then answer based on the results.
3. If you need to know which documents are available, call list_documents.
4. Cite the document filename and section titles when referencing information.
5. If a tool returns an error, explain it to the user in plain language.
6. If no documents are loaded, suggest the user upload one first.
7. When asked to summarize, classify, translate, or perform any Smart Tool operation,
   use run_smart_tool with the appropriate tool name.
8. When asked to generate a report, use generate_report.
9. When asked to process multiple documents at once, use run_batch.
10. For complex multi-document tasks, follow the Guided Workflow above.
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

        # Resolve document filenames from DB
        document_names: dict[str, str] = {}
        if document_ids and db:
            try:
                from idpkit.db.models import Document
                stmt = sa_select(Document.id, Document.filename).where(
                    Document.id.in_(document_ids)
                )
                rows = await db.execute(stmt)
                document_names = {r[0]: r[1] for r in rows}
            except Exception:
                pass  # Proceed without names

        # Record the user message
        conversation.add_message("user", message)

        # Build the messages list for the LLM
        messages = self._build_messages(conversation, document_ids, document_names)

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
        document_names: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build the full messages list including system prompt and history."""
        doc_context = ""
        if document_ids:
            names = document_names or {}
            doc_list = ", ".join(
                f"{did} ({names[did]})" if did in names else did
                for did in document_ids
            )
            doc_context = (
                f"\n\nThe user has selected these documents: {doc_list}. "
                f"You MUST search ALL of these documents before answering."
            )

        system_msg = {
            "role": "system",
            "content": self._system_prompt + doc_context,
        }

        # Get conversation history (only role + content for LLM compatibility)
        history = conversation.get_messages(limit=20)

        return [system_msg] + history
