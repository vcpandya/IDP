"""Smart Anonymize — Pseudonymization with consistent fake replacements."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartAnonymizeTool(BaseTool):
    """Pseudonymizes documents by replacing real entities with consistent fake ones."""

    @property
    def name(self) -> str:
        return "smart_anonymize"

    @property
    def display_name(self) -> str:
        return "Smart Anonymize"

    @property
    def description(self) -> str:
        return "Pseudonymize documents by replacing real entities with consistent, realistic fake replacements."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to anonymize",
                },
                "entity_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "person_names", "organizations", "locations",
                            "dates", "financial_amounts", "account_numbers",
                            "email_addresses", "phone_numbers",
                        ],
                    },
                    "default": ["person_names", "organizations", "locations", "email_addresses", "phone_numbers"],
                    "description": "Types of entities to anonymize",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible fake replacements (optional)",
                },
                "preserve_consistency": {
                    "type": "boolean",
                    "default": True,
                    "description": "Ensure the same real entity always maps to the same fake replacement throughout the document",
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
        entity_types = options.get("entity_types", [
            "person_names", "organizations", "locations", "email_addresses", "phone_numbers",
        ])
        seed = options.get("seed")
        preserve_consistency = options.get("preserve_consistency", True)

        seed_instruction = f"\nUse seed value {seed} to guide deterministic replacement choices." if seed is not None else ""
        consistency_instruction = (
            "\nIMPORTANT: The same real entity must always map to the same fake replacement throughout the entire document. "
            "For example, if 'John Smith' becomes 'Robert Johnson', every occurrence of 'John Smith' must become 'Robert Johnson'."
            if preserve_consistency else ""
        )

        prompt = f"""Anonymize the following document by replacing real entities with realistic fake replacements (pseudonymization).

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Entity types to anonymize: {json.dumps(entity_types)}
{seed_instruction}
{consistency_instruction}

Requirements:
- Replacements must be realistic and contextually appropriate
- Maintain the document's readability and logical flow
- Preserve data formats (e.g., phone numbers should look like phone numbers)
- Do not anonymize generic terms, only specific identifiable entities

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "entity_mapping": [
        {{
            "original": "John Smith",
            "replacement": "Robert Johnson",
            "entity_type": "person_names",
            "occurrences": 5
        }}
    ],
    "total_entities_found": <number>,
    "total_replacements_made": <number>,
    "entity_types_found": ["person_names", "organizations"],
    "anonymized_sections": [
        {{
            "section_title": "Section name",
            "anonymized_text": "The section text with entities replaced"
        }}
    ],
    "consistency_verified": true
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
                data={"raw_anonymization": response.content},
            )
        except Exception as e:
            logger.error("SmartAnonymize failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
