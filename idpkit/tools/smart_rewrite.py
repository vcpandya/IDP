"""Smart Rewrite — Content rewriting, simplification, and enrichment."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartRewriteTool(BaseTool):
    """Rewrites document content to improve clarity, simplify language, formalize tone, or enrich with context."""

    @property
    def name(self) -> str:
        return "smart_rewrite"

    @property
    def display_name(self) -> str:
        return "Smart Rewrite"

    @property
    def description(self) -> str:
        return "Rewrite document content to improve, simplify, formalize, or enrich it with adjustable tone and reading level."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to rewrite",
                },
                "mode": {
                    "type": "string",
                    "enum": ["improve", "simplify", "formalize", "enrich"],
                    "default": "improve",
                    "description": "Rewrite mode: improve (general quality), simplify (plain language), formalize (professional tone), enrich (add context and detail)",
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "casual", "academic", "friendly", "authoritative"],
                    "default": "professional",
                    "description": "Target tone for the rewritten content",
                },
                "reading_level": {
                    "type": "string",
                    "enum": ["elementary", "middle_school", "high_school", "college", "expert"],
                    "default": "college",
                    "description": "Target reading level for the output",
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
        mode = options.get("mode", "improve")
        tone = options.get("tone", "professional")
        reading_level = options.get("reading_level", "college")

        mode_instructions = {
            "improve": (
                "Improve the overall quality of the writing:\n"
                "- Fix grammar, spelling, and punctuation errors\n"
                "- Improve sentence structure and flow\n"
                "- Enhance clarity and conciseness\n"
                "- Strengthen word choices\n"
                "- Maintain the original meaning and intent"
            ),
            "simplify": (
                "Simplify the document into plain, accessible language:\n"
                "- Replace jargon and technical terms with simpler alternatives\n"
                "- Break complex sentences into shorter, clearer ones\n"
                "- Use active voice wherever possible\n"
                "- Add brief explanations for unavoidable technical concepts\n"
                "- Maintain accuracy while improving accessibility"
            ),
            "formalize": (
                "Formalize the document into professional, polished language:\n"
                "- Adopt a professional, authoritative tone\n"
                "- Replace informal language with formal equivalents\n"
                "- Ensure consistent formatting and structure\n"
                "- Use appropriate business/academic conventions\n"
                "- Remove colloquialisms and slang"
            ),
            "enrich": (
                "Enrich the document with additional context and detail:\n"
                "- Add relevant context and background information\n"
                "- Expand on key points with supporting details\n"
                "- Include transitional phrases for better flow\n"
                "- Suggest where examples, analogies, or data would strengthen the content\n"
                "- Maintain the original structure while enhancing depth"
            ),
        }

        reading_level_guidance = {
            "elementary": "Write at an elementary school level (grades 3-5). Use simple words and short sentences.",
            "middle_school": "Write at a middle school level (grades 6-8). Use clear language with moderate complexity.",
            "high_school": "Write at a high school level (grades 9-12). Use standard vocabulary and varied sentence structure.",
            "college": "Write at a college/university level. Use sophisticated vocabulary and complex ideas.",
            "expert": "Write at an expert/professional level. Use domain-specific terminology and assume advanced knowledge.",
        }

        prompt = f"""Rewrite the following document according to the specified parameters.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Rewrite Mode: {mode}
{mode_instructions.get(mode, mode_instructions['improve'])}

Target Tone: {tone}
Target Reading Level: {reading_level}
{reading_level_guidance.get(reading_level, reading_level_guidance['college'])}

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "rewrite_mode": "{mode}",
    "tone": "{tone}",
    "reading_level": "{reading_level}",
    "rewritten_sections": [
        {{
            "section_title": "Section name",
            "original_preview": "First 200 chars of original...",
            "rewritten_text": "The fully rewritten section text"
        }}
    ],
    "changes_summary": {{
        "total_sections_rewritten": <number>,
        "key_changes": ["description of major changes made"],
        "readability_improvement": "Brief assessment of readability improvement"
    }}
}}

Return ONLY the JSON, no other text."""

        try:
            response = await llm.acomplete(prompt)
            content = response.content.strip()
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
                data={"raw_rewrite": response.content},
            )
        except Exception as e:
            logger.error("SmartRewrite failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
