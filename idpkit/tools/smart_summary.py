"""Smart Summary — Hierarchical document summarization."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartSummaryTool(BaseTool):
    """Generates hierarchical summaries using the document tree structure."""

    @property
    def name(self) -> str:
        return "smart_summary"

    @property
    def display_name(self) -> str:
        return "Smart Summary"

    @property
    def description(self) -> str:
        return "Generate hierarchical summaries of documents with customizable length, style, and audience."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to summarize",
                },
                "length": {
                    "type": "string",
                    "enum": ["brief", "standard", "detailed"],
                    "default": "standard",
                    "description": "Summary length: brief (~1-2 paragraphs), standard (~3-5), detailed (comprehensive)",
                },
                "style": {
                    "type": "string",
                    "enum": ["executive", "technical", "bullets", "narrative"],
                    "default": "executive",
                    "description": "Summary style",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional focus area or topic to emphasize in the summary",
                },
                "highlight_key_messages": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to highlight key messages and takeaways",
                },
                "audience": {
                    "type": "string",
                    "description": "Target audience (e.g., 'C-suite', 'engineers', 'general public')",
                },
            },
            "required": ["document_id"],
        }

    async def execute(
        self,
        document_id: str,
        options: dict,
        llm: LLMClient,
        **kwargs: Any,
    ) -> ToolResult:
        db = kwargs.get("db")
        if not db:
            return ToolResult(tool_name=self.name, status="error", error="Database session required")

        from sqlalchemy import select
        from idpkit.db.models import Document

        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return ToolResult(tool_name=self.name, status="error", error="Document not found")

        tree_index = doc.tree_index or {}

        length = options.get("length", "standard")
        style = options.get("style", "executive")
        focus = options.get("focus", "")
        highlight_key = options.get("highlight_key_messages", True)
        audience = options.get("audience", "general")

        length_guidance = {
            "brief": "Provide a concise summary in 1-2 short paragraphs.",
            "standard": "Provide a thorough summary in 3-5 paragraphs covering all major points.",
            "detailed": "Provide a comprehensive, detailed summary covering every significant section and point.",
        }

        style_guidance = {
            "executive": "Write in executive summary style: clear, decisive, action-oriented. Lead with conclusions.",
            "technical": "Write in technical style: precise, detailed, use domain-specific terminology.",
            "bullets": "Format as bullet points organized by section/topic. Use sub-bullets for details.",
            "narrative": "Write in flowing narrative style, connecting ideas and providing context.",
        }

        focus_instruction = f"\nFocus especially on: {focus}" if focus else ""
        highlight_instruction = (
            "\nHighlight key messages, takeaways, and action items clearly."
            if highlight_key
            else ""
        )
        audience_instruction = f"\nTarget audience: {audience}. Adjust language and detail level accordingly."

        prompt = f"""Summarize the following document hierarchically, respecting its structure.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Instructions:
{length_guidance.get(length, length_guidance['standard'])}
{style_guidance.get(style, style_guidance['executive'])}
{focus_instruction}
{highlight_instruction}
{audience_instruction}

Return your response as valid JSON with this structure:
{{
    "title": "Summary title",
    "overall_summary": "High-level summary of the entire document",
    "section_summaries": [
        {{
            "section": "Section title",
            "summary": "Section summary"
        }}
    ],
    "key_takeaways": ["takeaway 1", "takeaway 2"],
    "key_messages": ["message 1", "message 2"]
}}

Return ONLY the JSON, no other text."""

        try:
            response = await llm.acomplete(prompt)
            content = response.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            data = json.loads(content)
            return ToolResult(
                tool_name=self.name,
                status="success",
                data=data,
            )
        except json.JSONDecodeError:
            return ToolResult(
                tool_name=self.name,
                status="success",
                data={"raw_summary": response.content},
            )
        except Exception as e:
            logger.error("SmartSummary failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
