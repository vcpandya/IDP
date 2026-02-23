"""Smart Redaction — PII detection and content redaction."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartRedactionTool(BaseTool):
    """Detects and redacts personally identifiable information (PII) from documents."""

    @property
    def name(self) -> str:
        return "smart_redaction"

    @property
    def display_name(self) -> str:
        return "Smart Redaction"

    @property
    def description(self) -> str:
        return "Detect and redact PII (names, emails, phone numbers, SSNs, etc.) from documents with audit logging."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to redact",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "selective"],
                    "default": "auto",
                    "description": "auto: detect and redact all PII; selective: only redact specified categories",
                },
                "pii_categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "names", "emails", "phone_numbers", "addresses",
                            "ssn", "credit_cards", "dates_of_birth", "ip_addresses",
                            "financial_accounts", "medical_ids",
                        ],
                    },
                    "default": ["names", "emails", "phone_numbers", "addresses", "ssn", "credit_cards"],
                    "description": "Categories of PII to detect and redact",
                },
                "redaction_style": {
                    "type": "string",
                    "enum": ["blackbox", "replaced", "hash"],
                    "default": "blackbox",
                    "description": "blackbox: replace with [REDACTED]; replaced: replace with type label [NAME], [EMAIL]; hash: replace with partial hash",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence score to redact (0.0-1.0)",
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
        mode = options.get("mode", "auto")
        pii_categories = options.get("pii_categories", [
            "names", "emails", "phone_numbers", "addresses", "ssn", "credit_cards",
        ])
        redaction_style = options.get("redaction_style", "blackbox")
        confidence_threshold = options.get("confidence_threshold", 0.7)

        style_instructions = {
            "blackbox": "Replace each PII instance with [REDACTED].",
            "replaced": "Replace each PII instance with a type label like [NAME], [EMAIL], [PHONE], [ADDRESS], [SSN], [CREDIT_CARD], etc.",
            "hash": "Replace each PII instance with a partial hash showing the type and first/last characters, e.g., N***n for a name, e***@***.com for email.",
        }

        category_text = ", ".join(pii_categories) if mode == "selective" else "all PII types"

        prompt = f"""Analyze the following document for personally identifiable information (PII) and produce a redacted version.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

PII Detection Mode: {mode}
Categories to detect: {category_text}
Confidence threshold: {confidence_threshold}
Redaction style: {redaction_style}
{style_instructions.get(redaction_style, style_instructions['blackbox'])}

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "redaction_style": "{redaction_style}",
    "pii_found": [
        {{
            "original": "the original PII text",
            "category": "names|emails|phone_numbers|addresses|ssn|credit_cards|dates_of_birth|ip_addresses|financial_accounts|medical_ids",
            "redacted": "the redacted replacement",
            "confidence": 0.95,
            "location": "section or context where found"
        }}
    ],
    "total_pii_detected": <number>,
    "total_redacted": <number>,
    "categories_found": ["list of PII categories detected"],
    "redacted_sections": [
        {{
            "section_title": "Section name",
            "redacted_text": "The section text with PII replaced"
        }}
    ],
    "audit_log": {{
        "scan_mode": "{mode}",
        "categories_scanned": {json.dumps(pii_categories)},
        "confidence_threshold": {confidence_threshold},
        "items_above_threshold": <number>,
        "items_below_threshold": <number>
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
                data={"raw_redaction": response.content},
            )
        except Exception as e:
            logger.error("SmartRedaction failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
