"""Smart Audit — Compliance, style, and completeness auditing."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartAuditTool(BaseTool):
    """Audits documents for compliance, style consistency, or completeness against standards."""

    @property
    def name(self) -> str:
        return "smart_audit"

    @property
    def display_name(self) -> str:
        return "Smart Audit"

    @property
    def description(self) -> str:
        return "Audit documents for compliance, style consistency, or completeness against standards and checklists."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to audit",
                },
                "audit_type": {
                    "type": "string",
                    "enum": ["compliance", "style", "completeness"],
                    "default": "completeness",
                    "description": "Type of audit: compliance (regulatory/policy), style (writing standards), completeness (coverage check)",
                },
                "checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom checklist items to audit against (e.g., ['Has executive summary', 'Includes risk analysis'])",
                },
                "reference_doc_id": {
                    "type": "string",
                    "description": "Optional reference document ID to audit against (e.g., a policy template or standard)",
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
        audit_type = options.get("audit_type", "completeness")
        checklist = options.get("checklist", [])
        reference_doc_id = options.get("reference_doc_id")

        # Optionally load reference document
        reference_text = ""
        if reference_doc_id:
            ref_result = await db.execute(select(Document).where(Document.id == reference_doc_id))
            ref_doc = ref_result.scalar_one_or_none()
            if ref_doc:
                ref_tree = ref_doc.tree_index or {}
                reference_text = f"\nReference Document: {ref_doc.filename}\nReference Tree Structure:\n{json.dumps(ref_tree, indent=2, default=str)}\n"
            else:
                reference_text = f"\nNote: Reference document '{reference_doc_id}' was not found.\n"

        checklist_text = ""
        if checklist:
            checklist_text = "\nCustom Checklist:\n" + "\n".join(f"- {item}" for item in checklist)

        type_instructions = {
            "compliance": (
                "Audit the document for regulatory and policy compliance:\n"
                "- Check for required legal/regulatory clauses\n"
                "- Verify mandatory disclosures and disclaimers\n"
                "- Identify missing compliance elements\n"
                "- Flag potential regulatory risks\n"
                "- Check for proper authorization and signature blocks"
            ),
            "style": (
                "Audit the document for writing style and consistency:\n"
                "- Check for consistent terminology throughout\n"
                "- Verify heading hierarchy and formatting consistency\n"
                "- Identify tone inconsistencies\n"
                "- Check for grammar, spelling, and punctuation issues\n"
                "- Evaluate clarity and readability\n"
                "- Flag jargon or undefined acronyms"
            ),
            "completeness": (
                "Audit the document for content completeness:\n"
                "- Identify expected sections that are missing\n"
                "- Check for incomplete sections or placeholder content\n"
                "- Verify all referenced items (figures, tables, appendices) exist\n"
                "- Check for dangling references or broken cross-references\n"
                "- Identify gaps in the logical flow or argumentation"
            ),
        }

        prompt = f"""Audit the following document according to the specified criteria.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Audit Type: {audit_type}
{type_instructions.get(audit_type, type_instructions['completeness'])}
{reference_text}
{checklist_text}

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "audit_type": "{audit_type}",
    "overall_score": 85,
    "overall_status": "pass|partial|fail",
    "summary": "High-level audit summary",
    "findings": [
        {{
            "id": 1,
            "severity": "critical|major|minor|info",
            "category": "category of the finding",
            "title": "Short finding title",
            "description": "Detailed description of the finding",
            "location": "Section or area in the document",
            "recommendation": "What should be done to address this"
        }}
    ],
    "checklist_results": [
        {{
            "item": "Checklist item text",
            "status": "pass|fail|partial|not_applicable",
            "notes": "Additional context"
        }}
    ],
    "statistics": {{
        "total_findings": <number>,
        "critical": <number>,
        "major": <number>,
        "minor": <number>,
        "info": <number>,
        "checklist_pass_rate": "<percentage>"
    }},
    "recommendations": ["Top priority recommendations"]
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
                data={"raw_audit": response.content},
            )
        except Exception as e:
            logger.error("SmartAudit failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
