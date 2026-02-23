"""Smart Merge — Multi-document merging and synthesis."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartMergeTool(BaseTool):
    """Merges multiple documents using concatenation, deduplication, or synthesis strategies."""

    @property
    def name(self) -> str:
        return "smart_merge"

    @property
    def display_name(self) -> str:
        return "Smart Merge"

    @property
    def description(self) -> str:
        return "Merge multiple documents by concatenating, deduplicating, or synthesizing their content."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Primary document ID (first document in the merge)",
                },
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of additional document IDs to merge with the primary document",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["concatenate", "deduplicate", "synthesize"],
                    "default": "concatenate",
                    "description": "Merge strategy: concatenate (append in order), deduplicate (remove overlapping content), synthesize (create unified narrative)",
                },
            },
            "required": ["document_id", "doc_ids"],
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

        doc_ids = options.get("doc_ids", [])
        if not doc_ids:
            return ToolResult(tool_name=self.name, status="error", error="doc_ids list is required")

        # Collect all document IDs (primary + additional)
        all_ids = [document_id] + doc_ids
        strategy = options.get("strategy", "concatenate")

        # Fetch all documents
        documents = []
        for did in all_ids:
            result = await db.execute(select(Document).where(Document.id == did))
            doc = result.scalar_one_or_none()
            if not doc:
                return ToolResult(tool_name=self.name, status="error", error=f"Document '{did}' not found")
            documents.append(doc)

        # Build document representations for the prompt
        doc_sections = []
        for doc in documents:
            tree = doc.tree_index or {}
            doc_sections.append({
                "filename": doc.filename,
                "document_id": doc.id,
                "description": doc.description or "N/A",
                "tree": tree,
            })

        strategy_instructions = {
            "concatenate": (
                "Concatenate the documents in order. Create a unified table of contents "
                "and merge sections sequentially. Preserve all content from all documents."
            ),
            "deduplicate": (
                "Merge the documents while removing duplicate or overlapping content. "
                "When the same topic is covered in multiple documents, keep the most "
                "comprehensive version. Note which sections were deduplicated."
            ),
            "synthesize": (
                "Synthesize the documents into a single coherent narrative. "
                "Reorganize content by topic rather than by source document. "
                "Combine related information, resolve contradictions, and create "
                "a unified document that reads as if written as a single piece."
            ),
        }

        documents_text = ""
        for i, ds in enumerate(doc_sections, 1):
            documents_text += f"\n--- Document {i}: {ds['filename']} ---\n"
            documents_text += f"Description: {ds['description']}\n"
            documents_text += f"Tree Structure:\n{json.dumps(ds['tree'], indent=2, default=str)}\n"

        prompt = f"""Merge the following {len(documents)} documents using the specified strategy.

{documents_text}

Merge Strategy: {strategy}
{strategy_instructions.get(strategy, strategy_instructions['concatenate'])}

Return your response as valid JSON with this structure:
{{
    "strategy": "{strategy}",
    "source_documents": [
        {{"filename": "name", "document_id": "id"}}
    ],
    "merged_outline": [
        {{
            "section_title": "Section name",
            "source_documents": ["filename(s) contributing to this section"],
            "content_preview": "Brief preview of merged content"
        }}
    ],
    "merged_text": "The full merged document text with proper headings and structure",
    "merge_notes": {{
        "total_sections": <number>,
        "sections_from_each": {{"filename": <count>}},
        "duplicates_removed": <number if deduplicate, 0 otherwise>,
        "conflicts_resolved": ["description of any conflicts resolved"]
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
                data={"raw_merge": response.content},
            )
        except Exception as e:
            logger.error("SmartMerge failed for documents %s: %s", all_ids, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
