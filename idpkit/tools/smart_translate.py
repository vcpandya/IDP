"""Smart Translate — Structure-preserving document translation."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartTranslateTool(BaseTool):
    """Translates document content while preserving structure, formatting, and meaning."""

    @property
    def name(self) -> str:
        return "smart_translate"

    @property
    def display_name(self) -> str:
        return "Smart Translate"

    @property
    def description(self) -> str:
        return "Translate documents between languages while preserving structure, formatting, and domain terminology."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to translate",
                },
                "target_language": {
                    "type": "string",
                    "description": "Target language for translation (e.g., 'Spanish', 'French', 'Japanese', 'de', 'zh')",
                },
                "source_language": {
                    "type": "string",
                    "default": "auto",
                    "description": "Source language (default: auto-detect)",
                },
                "preserve_formatting": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to preserve document formatting, headings, lists, and structure",
                },
            },
            "required": ["document_id", "target_language"],
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

        target_language = options.get("target_language")
        if not target_language:
            return ToolResult(tool_name=self.name, status="error", error="target_language is required")

        tree_index = doc.tree_index or {}
        source_language = options.get("source_language", "auto")
        preserve_formatting = options.get("preserve_formatting", True)

        source_instruction = (
            f"Source language: {source_language}"
            if source_language != "auto"
            else "Auto-detect the source language."
        )
        formatting_instruction = (
            "IMPORTANT: Preserve all formatting including headings, lists, bold/italic markers, "
            "tables, and document structure. Translate content but keep structural elements intact."
            if preserve_formatting
            else "Focus on accurate translation. Formatting preservation is not required."
        )

        prompt = f"""Translate the following document from its source language to {target_language}.

Document: {doc.filename}
Description: {doc.description or 'N/A'}
{source_instruction}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

{formatting_instruction}

Translation guidelines:
- Maintain accuracy and natural fluency in the target language
- Preserve domain-specific terminology where appropriate (provide both original and translated term on first use)
- Keep proper nouns, brand names, and acronyms in their original form unless they have standard translations
- Translate section titles and headings as well as body text

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "source_language": "<detected or specified source language>",
    "target_language": "{target_language}",
    "translated_sections": [
        {{
            "original_title": "Original section title",
            "translated_title": "Translated section title",
            "translated_text": "The fully translated section text"
        }}
    ],
    "translation_notes": [
        {{
            "term": "original term",
            "translation": "translated term",
            "note": "explanation of translation choice if non-obvious"
        }}
    ],
    "total_sections_translated": <number>,
    "formatting_preserved": {str(preserve_formatting).lower()}
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
                data={"raw_translation": response.content},
            )
        except Exception as e:
            logger.error("SmartTranslate failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
