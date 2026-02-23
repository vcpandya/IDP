"""Smart Fill — Auto-fill templates with data extracted from source documents."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartFillTool(BaseTool):
    """Extracts data from a source document and fills template fields automatically."""

    @property
    def name(self) -> str:
        return "smart_fill"

    @property
    def display_name(self) -> str:
        return "Smart Fill"

    @property
    def description(self) -> str:
        return "Auto-fill templates by extracting relevant data from source documents and mapping to template fields."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Source document ID to extract data from",
                },
                "template_doc_id": {
                    "type": "string",
                    "description": "Template document ID with fields/placeholders to fill",
                },
                "field_mapping": {
                    "type": "object",
                    "description": "Optional explicit mapping of template fields to source data paths, e.g. {'company_name': 'header.organization'}",
                },
            },
            "required": ["document_id", "template_doc_id"],
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

        # Fetch source document
        result_src = await db.execute(select(Document).where(Document.id == document_id))
        doc_src = result_src.scalar_one_or_none()
        if not doc_src:
            return ToolResult(tool_name=self.name, status="error", error=f"Source document '{document_id}' not found")

        # Fetch template document
        template_doc_id = options.get("template_doc_id")
        if not template_doc_id:
            return ToolResult(tool_name=self.name, status="error", error="template_doc_id is required")

        result_tpl = await db.execute(select(Document).where(Document.id == template_doc_id))
        doc_tpl = result_tpl.scalar_one_or_none()
        if not doc_tpl:
            return ToolResult(tool_name=self.name, status="error", error=f"Template document '{template_doc_id}' not found")

        tree_src = doc_src.tree_index or {}
        tree_tpl = doc_tpl.tree_index or {}
        field_mapping = options.get("field_mapping", {})

        mapping_instruction = ""
        if field_mapping:
            mapping_instruction = f"\nExplicit field mapping provided:\n{json.dumps(field_mapping, indent=2)}\nUse these mappings where possible, and auto-detect any remaining fields."

        prompt = f"""You are given a source document and a template document. Extract relevant data from the source document and fill the template's fields/placeholders.

SOURCE DOCUMENT: {doc_src.filename}
Description: {doc_src.description or 'N/A'}
Tree Structure:
{json.dumps(tree_src, indent=2, default=str)}

TEMPLATE DOCUMENT: {doc_tpl.filename}
Description: {doc_tpl.description or 'N/A'}
Tree Structure:
{json.dumps(tree_tpl, indent=2, default=str)}
{mapping_instruction}

Instructions:
1. Identify all fields, placeholders, or blanks in the template document.
2. For each field, search the source document for the matching data.
3. Fill each field with the extracted value.
4. Report any fields that could not be filled.

Return your response as valid JSON with this structure:
{{
    "source_document": "{doc_src.filename}",
    "template_document": "{doc_tpl.filename}",
    "filled_fields": [
        {{
            "field_name": "Field or placeholder name",
            "extracted_value": "Value from source document",
            "source_section": "Section in source where value was found",
            "confidence": 0.95
        }}
    ],
    "unfilled_fields": [
        {{
            "field_name": "Field name",
            "reason": "Why it could not be filled"
        }}
    ],
    "total_fields": <number>,
    "fields_filled": <number>,
    "fields_unfilled": <number>,
    "filled_template_text": "The template text with all fields filled in"
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
                data={"raw_fill": response.content},
            )
        except Exception as e:
            logger.error("SmartFill failed for document %s -> template %s: %s", document_id, template_doc_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
