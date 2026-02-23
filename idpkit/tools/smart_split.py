"""Smart Split — Structure-aware document splitting into chunks."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


def _flatten_tree(tree: Any) -> list[dict]:
    """Recursively flatten a tree index into a list of node dicts."""
    nodes: list[dict] = []
    if isinstance(tree, dict):
        node_copy = {k: v for k, v in tree.items() if k != "nodes"}
        nodes.append(node_copy)
        for child in tree.get("nodes", []):
            nodes.extend(_flatten_tree(child))
    elif isinstance(tree, list):
        for item in tree:
            nodes.extend(_flatten_tree(item))
    return nodes


class SmartSplitTool(BaseTool):
    """Splits documents into meaningful chunks using tree structure awareness."""

    @property
    def name(self) -> str:
        return "smart_split"

    @property
    def display_name(self) -> str:
        return "Smart Split"

    @property
    def description(self) -> str:
        return "Split documents into structure-aware chunks by sections, topics, or size constraints."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to split",
                },
                "split_by": {
                    "type": "string",
                    "enum": ["sections", "topics", "size"],
                    "default": "sections",
                    "description": "Split strategy: sections (follow tree hierarchy), topics (semantic grouping), size (fixed token budget)",
                },
                "max_chunk_size": {
                    "type": "integer",
                    "default": 2000,
                    "minimum": 100,
                    "maximum": 32000,
                    "description": "Maximum chunk size in approximate tokens (used for size-based splitting)",
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
        split_by = options.get("split_by", "sections")
        max_chunk_size = options.get("max_chunk_size", 2000)

        if split_by == "sections":
            # Direct tree-based splitting without LLM
            chunks = self._split_by_sections(tree_index, doc.filename)
            return ToolResult(
                tool_name=self.name,
                status="success",
                data={
                    "document": doc.filename,
                    "split_strategy": "sections",
                    "total_chunks": len(chunks),
                    "chunks": chunks,
                },
            )

        if split_by == "size":
            # Size-constrained splitting using flattened nodes
            chunks = self._split_by_size(tree_index, doc.filename, max_chunk_size)
            return ToolResult(
                tool_name=self.name,
                status="success",
                data={
                    "document": doc.filename,
                    "split_strategy": "size",
                    "max_chunk_size": max_chunk_size,
                    "total_chunks": len(chunks),
                    "chunks": chunks,
                },
            )

        # Topic-based splitting requires LLM
        prompt = f"""Analyze the following document structure and split it into coherent topical chunks.
Group sections that discuss the same topic together, even if they are not adjacent in the document.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "split_strategy": "topics",
    "total_chunks": <number>,
    "chunks": [
        {{
            "chunk_id": 1,
            "topic": "Topic name",
            "node_ids": ["node_id_1", "node_id_2"],
            "section_titles": ["Section A", "Section B"],
            "summary": "Brief summary of this topic chunk",
            "estimated_tokens": <approximate token count>
        }}
    ]
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
                data={"raw_split": response.content},
            )
        except Exception as e:
            logger.error("SmartSplit failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))

    def _split_by_sections(self, tree_index: dict, filename: str) -> list[dict]:
        """Split by tree hierarchy — each top-level node becomes a chunk."""
        chunks = []
        structure = tree_index.get("structure", [])
        if not structure and isinstance(tree_index, dict):
            structure = tree_index.get("nodes", [])

        for idx, node in enumerate(structure, 1):
            text = node.get("text", "")
            summary = node.get("summary") or node.get("prefix_summary") or ""
            chunks.append({
                "chunk_id": idx,
                "node_id": node.get("node_id"),
                "title": node.get("title", f"Section {idx}"),
                "text_preview": text[:500] if text else summary[:500],
                "estimated_tokens": len(text.split()) if text else 0,
                "child_count": len(node.get("nodes", [])),
            })
        return chunks

    def _split_by_size(self, tree_index: dict, filename: str, max_tokens: int) -> list[dict]:
        """Split into size-constrained chunks respecting node boundaries."""
        all_nodes = _flatten_tree(tree_index.get("structure", tree_index))
        chunks = []
        current_chunk: list[dict] = []
        current_size = 0
        chunk_id = 1

        for node in all_nodes:
            text = node.get("text", "")
            node_tokens = len(text.split()) if text else 0

            if current_size + node_tokens > max_tokens and current_chunk:
                # Flush current chunk
                chunks.append({
                    "chunk_id": chunk_id,
                    "sections": [n.get("title", "") for n in current_chunk],
                    "node_ids": [n.get("node_id") for n in current_chunk if n.get("node_id")],
                    "estimated_tokens": current_size,
                })
                chunk_id += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(node)
            current_size += node_tokens

        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "sections": [n.get("title", "") for n in current_chunk],
                "node_ids": [n.get("node_id") for n in current_chunk if n.get("node_id")],
                "estimated_tokens": current_size,
            })

        return chunks
