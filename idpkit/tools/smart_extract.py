"""Smart Extract — Structured data extraction from documents."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartExtractTool(BaseTool):
    """Extracts structured data such as tables, key-value pairs, entities, and financials."""

    @property
    def name(self) -> str:
        return "smart_extract"

    @property
    def display_name(self) -> str:
        return "Smart Extract"

    @property
    def description(self) -> str:
        return "Extract structured data (tables, key-value pairs, entities, financial data) from documents."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to extract from",
                },
                "type": {
                    "type": "string",
                    "enum": ["tables", "key_value", "entities", "financial", "custom"],
                    "default": "key_value",
                    "description": "Type of structured data to extract",
                },
                "custom_schema": {
                    "type": "object",
                    "description": "Custom extraction schema when type is 'custom'. Defines fields to extract.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "markdown"],
                    "default": "json",
                    "description": "Output format for extracted data",
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
        extract_type = options.get("type", "key_value")
        custom_schema = options.get("custom_schema", {})
        output_format = options.get("output_format", "json")

        type_instructions = {
            "tables": (
                "Extract all tabular data found in the document. For each table, provide:\n"
                "- table_name: descriptive name for the table\n"
                "- headers: list of column headers\n"
                "- rows: list of row data (each row is a list of cell values)\n"
                "- location: section where the table was found"
            ),
            "key_value": (
                "Extract all key-value pairs from the document. For each pair, provide:\n"
                "- key: the field name/label\n"
                "- value: the extracted value\n"
                "- confidence: how confident you are (0-1)\n"
                "- source_section: where in the document this was found"
            ),
            "entities": (
                "Extract named entities from the document. For each entity, provide:\n"
                "- text: the entity text\n"
                "- type: entity type (PERSON, ORGANIZATION, LOCATION, DATE, AMOUNT, etc.)\n"
                "- context: brief surrounding context\n"
                "- section: where in the document this was found"
            ),
            "financial": (
                "Extract all financial data from the document. Include:\n"
                "- amounts: list of monetary amounts with labels and currency\n"
                "- dates: relevant financial dates\n"
                "- accounts: account numbers or references\n"
                "- totals: any totals, subtotals, or aggregated values\n"
                "- terms: payment terms, interest rates, etc."
            ),
            "custom": (
                f"Extract data according to this custom schema:\n{json.dumps(custom_schema, indent=2)}\n"
                "Return data matching the schema fields exactly."
            ),
        }

        format_instructions = {
            "json": "Return the extracted data as valid JSON.",
            "csv": "Return the extracted data in CSV format with headers.",
            "markdown": "Return the extracted data formatted as markdown tables.",
        }

        prompt = f"""Extract structured data from the following document.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Extraction Type: {extract_type}
{type_instructions.get(extract_type, type_instructions['key_value'])}

Output Format: {output_format}
{format_instructions.get(output_format, format_instructions['json'])}

Return your response as valid JSON with this structure:
{{
    "extraction_type": "{extract_type}",
    "item_count": <number of items extracted>,
    "data": <extracted data according to the type instructions above>,
    "metadata": {{
        "sections_processed": <number>,
        "extraction_notes": "any relevant notes about the extraction"
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
                data={"raw_extraction": response.content},
            )
        except Exception as e:
            logger.error("SmartExtract failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
