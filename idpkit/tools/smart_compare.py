"""Smart Compare — Document comparison and diff analysis."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartCompareTool(BaseTool):
    """Compares two documents structurally, semantically, or for regulatory alignment."""

    @property
    def name(self) -> str:
        return "smart_compare"

    @property
    def display_name(self) -> str:
        return "Smart Compare"

    @property
    def description(self) -> str:
        return "Compare two documents for structural, semantic, or regulatory differences."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Primary document ID to compare",
                },
                "compare_doc_id": {
                    "type": "string",
                    "description": "Second document ID to compare against",
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["structural", "semantic", "regulatory"],
                    "default": "semantic",
                    "description": "Type of comparison: structural (outline/headings), semantic (meaning/content), regulatory (compliance alignment)",
                },
            },
            "required": ["document_id", "compare_doc_id"],
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

        compare_doc_id = options.get("compare_doc_id")
        if not compare_doc_id:
            return ToolResult(tool_name=self.name, status="error", error="compare_doc_id is required")

        # Fetch both documents
        result_a = await db.execute(select(Document).where(Document.id == document_id))
        doc_a = result_a.scalar_one_or_none()
        if not doc_a:
            return ToolResult(tool_name=self.name, status="error", error=f"Document '{document_id}' not found")

        result_b = await db.execute(select(Document).where(Document.id == compare_doc_id))
        doc_b = result_b.scalar_one_or_none()
        if not doc_b:
            return ToolResult(tool_name=self.name, status="error", error=f"Document '{compare_doc_id}' not found")

        tree_a = doc_a.tree_index or {}
        tree_b = doc_b.tree_index or {}
        comparison_type = options.get("comparison_type", "semantic")

        type_instructions = {
            "structural": (
                "Compare the structural organization of these two documents:\n"
                "- Heading hierarchy and depth\n"
                "- Section ordering and naming patterns\n"
                "- Number of sections and subsections\n"
                "- Structural elements present in one but missing in the other"
            ),
            "semantic": (
                "Compare the content and meaning of these two documents:\n"
                "- Key topics and themes in common\n"
                "- Unique content in each document\n"
                "- Contradictions or conflicting statements\n"
                "- Complementary information that adds context"
            ),
            "regulatory": (
                "Compare these documents from a regulatory/compliance perspective:\n"
                "- Clauses or requirements present in one but missing in the other\n"
                "- Differences in obligations, deadlines, or thresholds\n"
                "- Changes in scope, definitions, or applicability\n"
                "- Risk implications of the differences"
            ),
        }

        prompt = f"""Compare the following two documents.

Document A: {doc_a.filename}
Description: {doc_a.description or 'N/A'}
Tree Structure:
{json.dumps(tree_a, indent=2, default=str)}

Document B: {doc_b.filename}
Description: {doc_b.description or 'N/A'}
Tree Structure:
{json.dumps(tree_b, indent=2, default=str)}

Comparison Type: {comparison_type}
{type_instructions.get(comparison_type, type_instructions['semantic'])}

Return your response as valid JSON with this structure:
{{
    "comparison_type": "{comparison_type}",
    "document_a": "{doc_a.filename}",
    "document_b": "{doc_b.filename}",
    "similarity_score": 0.75,
    "summary": "High-level comparison summary",
    "similarities": [
        {{"aspect": "...", "description": "..."}}
    ],
    "differences": [
        {{"aspect": "...", "in_doc_a": "...", "in_doc_b": "...", "significance": "high|medium|low"}}
    ],
    "unique_to_a": ["items only in document A"],
    "unique_to_b": ["items only in document B"],
    "recommendations": ["actionable recommendations based on the comparison"]
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
                data={"raw_comparison": response.content},
            )
        except Exception as e:
            logger.error("SmartCompare failed for documents %s vs %s: %s", document_id, compare_doc_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
