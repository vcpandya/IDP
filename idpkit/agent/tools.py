"""IDP Kit Agent tool definitions and executors.

Each tool is exposed as:
- A dict in OpenAI function-calling format (``TOOL_DEFINITIONS``).
- An async ``execute(args, llm, db)`` implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from idpkit.db.models import Document

logger = logging.getLogger(__name__)


# ======================================================================
# Tool JSON Schema definitions (OpenAI function-calling format)
# ======================================================================

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search a document's tree index for sections relevant to a query. "
                "Returns matching nodes with titles, summaries, and text snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The ID of the document to search.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query describing what to find.",
                    },
                },
                "required": ["document_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all available documents the user has access to, "
                "including their IDs, filenames, formats, and indexing status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_section",
            "description": (
                "Summarize a specific section (node) of a document identified "
                "by document_id and node_id.  Uses the LLM to produce a concise summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The ID of the document.",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "The node_id of the section to summarize.",
                    },
                },
                "required": ["document_id", "node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_data",
            "description": (
                "Extract structured data of a given type from a document. "
                "Supported data types include: tables, figures, key_facts, "
                "financial, dates, entities, and custom types described in natural language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The ID of the document to extract data from.",
                    },
                    "data_type": {
                        "type": "string",
                        "description": (
                            "The type of data to extract.  Examples: 'tables', "
                            "'figures', 'key_facts', 'financial', 'dates', 'entities', "
                            "or a free-form description."
                        ),
                    },
                },
                "required": ["document_id", "data_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_graph",
            "description": (
                "Query the knowledge graph for entity information. Supports "
                "operations: find_entity, entity_mentions, related_sections, "
                "cross_document_links, document_entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "find_entity",
                            "entity_mentions",
                            "related_sections",
                            "cross_document_links",
                            "document_entities",
                        ],
                        "description": "The graph query operation to perform.",
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to search for (for find_entity).",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID (for entity_mentions).",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Document ID (for document_entities, cross_document_links, related_sections).",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "Node ID (for related_sections).",
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cross_references",
            "description": (
                "Find all sections across all documents that mention a given "
                "topic or entity. Returns document names, section titles, and "
                "page references for every occurrence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or entity name to find cross-references for.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_smart_tool",
            "description": (
                "Execute any of IDP Kit's 13 Smart Tools on a document. "
                "Available tools: smart_summary, smart_classify, smart_extract, "
                "smart_compare, smart_qa, smart_split, smart_redaction, "
                "smart_anonymize, smart_fill, smart_rewrite, smart_translate, "
                "smart_merge, smart_audit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "enum": [
                            "smart_summary", "smart_classify", "smart_extract",
                            "smart_compare", "smart_qa", "smart_split",
                            "smart_redaction", "smart_anonymize", "smart_fill",
                            "smart_rewrite", "smart_translate", "smart_merge",
                            "smart_audit",
                        ],
                        "description": "The Smart Tool to execute.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The ID of the document to process.",
                    },
                    "options": {
                        "type": "object",
                        "description": "Tool-specific options (e.g. language for translate, style for summary).",
                    },
                },
                "required": ["tool_name", "document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Generate a structured report from an indexed document in "
                "Markdown or DOCX format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The ID of the document to generate a report from.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "docx"],
                        "description": "Output format (default: markdown).",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_batch",
            "description": (
                "Create a batch processing job to run a Smart Tool on multiple "
                "documents. The batch runs in the background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The Smart Tool to run on each document.",
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to process.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional prompt or instructions for the tool.",
                    },
                    "options": {
                        "type": "object",
                        "description": "Tool-specific options applied to all documents.",
                    },
                },
                "required": ["tool_name", "document_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compose_with_context",
            "description": (
                "Compose a document or response using multiple documents in different roles. "
                "Use this when the user needs to: draft a response to a document, create a "
                "deliverable using reference material, analyze documents against a template, "
                "or any task where documents serve different purposes (primary task, supporting "
                "context, format reference). Examples: draft a reply to a legal notice using "
                "case files as context and a past response as format reference; write a report "
                "from research data with a template; compare a proposal against requirements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to produce — e.g., 'Draft a response', 'Create a summary report', 'Analyze compliance'.",
                    },
                    "primary_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Main document(s) to work on — the notice to respond to, the paper to review, the contract to analyze.",
                    },
                    "context_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Supporting documents for background/evidence — case files, internal memos, reference data.",
                    },
                    "reference_doc_id": {
                        "type": "string",
                        "description": "Optional: a document showing the desired output format/structure — a past response, a template, a sample report.",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Additional user instructions — tone, focus areas, what to include/exclude.",
                    },
                },
                "required": ["task", "primary_doc_ids"],
            },
        },
    },
]

# Quick lookup by name
_TOOL_MAP: dict[str, dict] = {
    t["function"]["name"]: t for t in TOOL_DEFINITIONS
}


# ======================================================================
# Helper: traverse a tree index
# ======================================================================

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


def _find_node_by_id(tree: Any, node_id: str) -> dict | None:
    """Find a specific node in the tree by ``node_id``."""
    if isinstance(tree, dict):
        if tree.get("node_id") == node_id:
            return tree
        for child in tree.get("nodes", []):
            found = _find_node_by_id(child, node_id)
            if found:
                return found
    elif isinstance(tree, list):
        for item in tree:
            found = _find_node_by_id(item, node_id)
            if found:
                return found
    return None


async def _get_document(db: AsyncSession, document_id: str) -> Document | None:
    """Fetch a document from the database."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


# ======================================================================
# Tool execution functions
# ======================================================================

async def _execute_search_document(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Search a document's tree index for relevant sections."""
    document_id = args["document_id"]
    query = args["query"]

    doc = await _get_document(db, document_id)
    if not doc:
        return {"error": f"Document '{document_id}' not found."}

    if not doc.tree_index:
        return {
            "error": f"Document '{doc.filename}' has not been indexed yet.",
            "document_id": document_id,
        }

    # Flatten the tree and let the LLM pick the best matches
    all_nodes = _flatten_tree(doc.tree_index)

    # Build a compact representation for the LLM to rank
    nodes_summary = []
    for node in all_nodes:
        entry = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "summary": node.get("summary") or node.get("prefix_summary") or "",
        }
        nodes_summary.append(entry)

    ranking_prompt = (
        f"Given the following document sections and the user query, "
        f"return a JSON array of the top relevant node_ids (max 5) "
        f"that best answer the query.  Return ONLY a JSON array of strings.\n\n"
        f"Query: {query}\n\n"
        f"Sections:\n{json.dumps(nodes_summary, indent=2)}"
    )

    try:
        response = await llm.acomplete(ranking_prompt)
        # Parse the LLM response to get relevant node IDs
        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        relevant_ids = json.loads(content)
    except Exception:
        # Fallback: return all nodes
        relevant_ids = [n.get("node_id") for n in all_nodes[:5]]

    # Gather full data for matched nodes
    results = []
    for nid in relevant_ids:
        for node in all_nodes:
            if node.get("node_id") == nid:
                results.append({
                    "node_id": nid,
                    "title": node.get("title"),
                    "summary": node.get("summary") or node.get("prefix_summary") or "",
                    "text_preview": (node.get("text") or "")[:500],
                    "start_page": node.get("start_index"),
                    "end_page": node.get("end_index"),
                })
                break

    return {
        "document_id": document_id,
        "filename": doc.filename,
        "query": query,
        "results": results,
    }


async def _execute_list_documents(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """List all documents in the database."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).limit(50)
    )
    docs = result.scalars().all()

    doc_list = []
    for doc in docs:
        doc_list.append({
            "id": doc.id,
            "filename": doc.filename,
            "format": doc.format,
            "status": doc.status,
            "description": doc.description,
            "page_count": doc.page_count,
        })

    return {"documents": doc_list, "total": len(doc_list)}


async def _execute_summarize_section(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Summarize a specific section of a document."""
    document_id = args["document_id"]
    node_id = args["node_id"]

    doc = await _get_document(db, document_id)
    if not doc:
        return {"error": f"Document '{document_id}' not found."}

    if not doc.tree_index:
        return {"error": f"Document '{doc.filename}' has not been indexed yet."}

    node = _find_node_by_id(doc.tree_index, node_id)
    if not node:
        return {"error": f"Node '{node_id}' not found in document '{doc.filename}'."}

    # If the node already has a summary, return it directly
    existing_summary = node.get("summary") or node.get("prefix_summary")
    text = node.get("text", "")

    if not text and existing_summary:
        return {
            "document_id": document_id,
            "node_id": node_id,
            "title": node.get("title"),
            "summary": existing_summary,
            "source": "cached",
        }

    # Generate a new summary via LLM
    section_text = text[:8000] if text else f"Title: {node.get('title')}"
    prompt = (
        f"Summarize the following document section concisely.  "
        f"Focus on the key points and main findings.\n\n"
        f"Section title: {node.get('title')}\n"
        f"Section text:\n{section_text}"
    )

    try:
        response = await llm.acomplete(prompt)
        summary = response.content
    except Exception as exc:
        logger.error("LLM summarization failed: %s", exc)
        summary = existing_summary or "Failed to generate summary."

    return {
        "document_id": document_id,
        "node_id": node_id,
        "title": node.get("title"),
        "summary": summary,
        "source": "generated",
    }


async def _execute_extract_data(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Extract structured data from a document."""
    document_id = args["document_id"]
    data_type = args["data_type"]

    doc = await _get_document(db, document_id)
    if not doc:
        return {"error": f"Document '{document_id}' not found."}

    if not doc.tree_index:
        return {"error": f"Document '{doc.filename}' has not been indexed yet."}

    # Collect all text from the tree
    all_nodes = _flatten_tree(doc.tree_index)
    full_text_parts = []
    for node in all_nodes:
        text = node.get("text", "")
        if text:
            title = node.get("title", "")
            full_text_parts.append(f"## {title}\n{text}")

    # Truncate to a reasonable context length
    combined_text = "\n\n".join(full_text_parts)
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000] + "\n\n[... truncated ...]"

    prompt = (
        f"Extract all {data_type} from the following document text.  "
        f"Return the results as a JSON object with a key 'extracted' "
        f"containing a list of items.  Each item should have relevant "
        f"fields depending on the data type.\n\n"
        f"Document: {doc.filename}\n"
        f"Data type to extract: {data_type}\n\n"
        f"Document text:\n{combined_text}"
    )

    try:
        response = await llm.acomplete(prompt)
        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        extracted = json.loads(content)
    except json.JSONDecodeError:
        extracted = {"raw_response": response.content}
    except Exception as exc:
        logger.error("LLM data extraction failed: %s", exc)
        extracted = {"error": f"Extraction failed: {exc}"}

    return {
        "document_id": document_id,
        "filename": doc.filename,
        "data_type": data_type,
        "extracted": extracted,
    }


# ======================================================================
# Knowledge Graph tool executors
# ======================================================================

def _validate_str_arg(args: dict, key: str, max_len: int = 500) -> str:
    """Extract and validate a string argument from tool args."""
    val = args.get(key, "")
    if isinstance(val, str):
        return val[:max_len]
    return str(val)[:max_len] if val else ""


async def _execute_query_graph(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Execute a knowledge graph query operation."""
    operation = _validate_str_arg(args, "operation", 50)

    try:
        if operation == "find_entity":
            from idpkit.graph.queries import search_entities
            name = _validate_str_arg(args, "entity_name", 200)
            if not name:
                return {"error": "entity_name is required for find_entity operation."}
            entities = await search_entities(db, name=name, limit=10)
            return {
                "operation": operation,
                "results": [
                    {
                        "id": e.id,
                        "name": e.canonical_name,
                        "type": e.entity_type,
                        "description": e.description,
                        "document_count": e.document_count,
                    }
                    for e in entities
                ],
            }

        elif operation == "entity_mentions":
            from idpkit.graph.queries import get_entity_mentions
            entity_id = _validate_str_arg(args, "entity_id", 100)
            if not entity_id:
                return {"error": "entity_id is required for entity_mentions operation."}
            mentions = await get_entity_mentions(db, entity_id)
            return {
                "operation": operation,
                "entity_id": entity_id,
                "mentions": [
                    {
                        "document_id": m.document_id,
                        "node_id": m.node_id,
                        "node_title": m.node_title,
                        "mention_text": m.mention_text,
                        "start_page": m.start_page,
                        "end_page": m.end_page,
                    }
                    for m in mentions
                ],
            }

        elif operation == "related_sections":
            from idpkit.graph.queries import get_related_sections
            document_id = _validate_str_arg(args, "document_id", 100)
            node_id = _validate_str_arg(args, "node_id", 100)
            if not document_id or not node_id:
                return {"error": "document_id and node_id are required for related_sections."}
            sections = await get_related_sections(db, document_id, node_id)
            return {
                "operation": operation,
                "results": [
                    {
                        "document_id": s["document_id"],
                        "node_id": s["node_id"],
                        "node_title": s["node_title"],
                        "shared_entity_count": s["shared_entity_count"],
                    }
                    for s in sections[:10]
                ],
            }

        elif operation == "cross_document_links":
            from idpkit.graph.queries import get_cross_document_links
            document_id = _validate_str_arg(args, "document_id", 100)
            if not document_id:
                return {"error": "document_id is required for cross_document_links."}
            links = await get_cross_document_links(db, document_id)
            return {
                "operation": operation,
                "document_id": document_id,
                "linked_documents": [
                    {
                        "document_id": link["document_id"],
                        "filename": link["filename"],
                        "shared_entity_count": len(link["shared_entities"]),
                        "shared_entities": [
                            e.canonical_name for e in link["shared_entities"][:5]
                        ],
                    }
                    for link in links
                ],
            }

        elif operation == "document_entities":
            from idpkit.graph.queries import get_document_entities
            document_id = _validate_str_arg(args, "document_id", 100)
            if not document_id:
                return {"error": "document_id is required for document_entities."}
            entities = await get_document_entities(db, document_id)
            return {
                "operation": operation,
                "document_id": document_id,
                "entities": [
                    {
                        "id": e.id,
                        "name": e.canonical_name,
                        "type": e.entity_type,
                        "description": e.description,
                    }
                    for e in entities
                ],
            }

        else:
            return {"error": f"Unknown operation: {operation}"}

    except Exception as exc:
        logger.error("query_graph operation '%s' failed: %s", operation, exc)
        return {"error": f"Graph query failed: {exc}"}


async def _execute_find_cross_references(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Find all mentions of a topic across all documents."""
    topic = args.get("topic", "")
    if not topic or len(topic) > 500:
        return {"error": "topic is required (max 500 characters)."}

    try:
        from idpkit.graph.queries import search_entities, get_entity_mentions
        from idpkit.graph.models import EntityMention as EM

        # Find matching entities
        entities = await search_entities(db, name=topic[:200], limit=5)
        if not entities:
            return {
                "topic": topic,
                "found": False,
                "message": f"No entities matching '{topic}' found in the knowledge graph.",
            }

        # Batch-load all mentions for matching entities in one query
        entity_ids = [e.id for e in entities]
        entity_map = {e.id: e for e in entities}

        mentions_result = await db.execute(
            select(EM)
            .where(EM.entity_id.in_(entity_ids))
            .limit(200)
        )
        all_mentions = mentions_result.scalars().all()

        # Batch-load all referenced documents in one query
        doc_ids_needed = set(m.document_id for m in all_mentions)
        if doc_ids_needed:
            docs_result = await db.execute(
                select(Document).where(Document.id.in_(list(doc_ids_needed)))
            )
            doc_map = {d.id: d for d in docs_result.scalars().all()}
        else:
            doc_map = {}

        all_references: list[dict] = []
        for m in all_mentions:
            entity = entity_map.get(m.entity_id)
            doc = doc_map.get(m.document_id)
            all_references.append({
                "entity_name": entity.canonical_name if entity else "unknown",
                "entity_type": entity.entity_type if entity else "unknown",
                "document_id": m.document_id,
                "document_filename": doc.filename if doc else "unknown",
                "node_id": m.node_id,
                "node_title": m.node_title,
                "start_page": m.start_page,
                "end_page": m.end_page,
            })

        unique_doc_ids = set(r["document_id"] for r in all_references)
        return {
            "topic": topic,
            "found": True,
            "total_mentions": len(all_references),
            "document_count": len(unique_doc_ids),
            "references": all_references,
        }

    except Exception as exc:
        logger.error("find_cross_references failed: %s", exc)
        return {"error": f"Cross-reference search failed: {exc}"}


# ======================================================================
# IDA Enhanced tool executors
# ======================================================================

async def _execute_run_smart_tool(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Execute any of the 13 Smart Tools on a document."""
    tool_name = args.get("tool_name", "")
    document_id = args.get("document_id", "")
    options = args.get("options") or {}

    if not tool_name or not document_id:
        return {"error": "tool_name and document_id are required."}

    try:
        from idpkit.tools import TOOL_REGISTRY

        tool = TOOL_REGISTRY.get(tool_name)
        if not tool:
            available = ", ".join(sorted(TOOL_REGISTRY.keys()))
            return {"error": f"Unknown tool '{tool_name}'. Available: {available}"}

        result = await tool.execute(
            document_id=document_id,
            options=options,
            llm=llm,
            db=db,
        )

        result_dict = {
            "tool_name": result.tool_name,
            "status": result.status,
        }
        if result.error:
            result_dict["error"] = result.error
        if result.output_file:
            result_dict["output_file"] = result.output_file
        if result.data:
            # Truncate large data to fit in agent context
            data_str = json.dumps(result.data, default=str)
            if len(data_str) > 5000:
                result_dict["data_preview"] = data_str[:5000] + "... [truncated]"
                result_dict["data_truncated"] = True
            else:
                result_dict["data"] = result.data

        return result_dict

    except Exception as exc:
        logger.error("run_smart_tool '%s' failed: %s", tool_name, exc)
        return {"error": f"Smart tool execution failed: {exc}"}


async def _execute_generate_report(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Generate a report from an indexed document."""
    document_id = args.get("document_id", "")
    fmt = args.get("format", "markdown")

    if not document_id:
        return {"error": "document_id is required."}

    doc = await _get_document(db, document_id)
    if not doc:
        return {"error": f"Document '{document_id}' not found."}

    if not doc.tree_index:
        return {"error": f"Document '{doc.filename}' has not been indexed yet."}

    try:
        from idpkit.generation import generate_markdown

        md_content = generate_markdown(doc.tree_index)

        if fmt == "docx":
            from idpkit.generation import generate_docx
            import tempfile
            import os

            output_path = os.path.join(
                tempfile.gettempdir(),
                f"report_{doc.id[:8]}.docx",
            )
            generate_docx(md_content, output_path)
            return {
                "document_id": document_id,
                "filename": doc.filename,
                "format": "docx",
                "output_path": output_path,
                "message": f"DOCX report generated at {output_path}",
            }

        # Markdown output
        if len(md_content) > 5000:
            preview = md_content[:5000] + "\n\n... [truncated]"
        else:
            preview = md_content

        return {
            "document_id": document_id,
            "filename": doc.filename,
            "format": "markdown",
            "content": preview,
            "full_length": len(md_content),
        }

    except Exception as exc:
        logger.error("generate_report failed: %s", exc)
        return {"error": f"Report generation failed: {exc}"}


async def _execute_run_batch(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Create a batch processing job for multiple documents."""
    tool_name = args.get("tool_name", "")
    document_ids = args.get("document_ids") or []
    prompt = args.get("prompt")
    options = args.get("options") or {}

    if not tool_name or not document_ids:
        return {"error": "tool_name and document_ids are required."}

    try:
        from idpkit.db.models import BatchJob, BatchItem
        from idpkit.batch.runner import run_batch_job

        batch_id = str(uuid.uuid4())
        batch = BatchJob(
            id=batch_id,
            owner_id="agent",
            tool_name=tool_name,
            status="pending",
            total_items=len(document_ids),
            completed_items=0,
            failed_items=0,
            prompt=prompt,
            options=options,
        )
        db.add(batch)

        for idx, doc_id in enumerate(document_ids):
            item = BatchItem(
                id=str(uuid.uuid4()),
                batch_job_id=batch_id,
                document_id=doc_id,
                position=idx,
                status="pending",
            )
            db.add(item)

        await db.commit()

        # Launch background task
        asyncio.create_task(run_batch_job(batch_id))

        return {
            "batch_id": batch_id,
            "status": "pending",
            "tool_name": tool_name,
            "document_count": len(document_ids),
            "message": f"Batch job created with {len(document_ids)} documents. Processing in background.",
        }

    except Exception as exc:
        logger.error("run_batch failed: %s", exc)
        return {"error": f"Batch creation failed: {exc}"}


async def _execute_compose_with_context(
    args: dict, llm: LLMClient, db: AsyncSession
) -> dict:
    """Compose a document using multiple documents in different roles."""
    task = args.get("task", "")
    primary_doc_ids = args.get("primary_doc_ids") or []
    context_doc_ids = args.get("context_doc_ids") or []
    reference_doc_id = args.get("reference_doc_id")
    instructions = args.get("instructions", "")

    if not task or not primary_doc_ids:
        return {"error": "task and primary_doc_ids are required."}

    def _extract_text(tree: Any, max_chars: int = 6000) -> str:
        """Extract text from a tree index up to max_chars."""
        nodes = _flatten_tree(tree)
        parts = []
        total = 0
        for node in nodes:
            text = node.get("text", "")
            title = node.get("title", "")
            if text:
                chunk = f"## {title}\n{text}" if title else text
                if total + len(chunk) > max_chars:
                    remaining = max_chars - total
                    if remaining > 100:
                        parts.append(chunk[:remaining] + "...")
                    break
                parts.append(chunk)
                total += len(chunk)
        return "\n\n".join(parts)

    try:
        # Load primary documents
        primary_texts = []
        for doc_id in primary_doc_ids:
            doc = await _get_document(db, doc_id)
            if not doc:
                return {"error": f"Primary document '{doc_id}' not found."}
            if not doc.tree_index:
                return {"error": f"Document '{doc.filename}' has not been indexed yet."}
            text = _extract_text(doc.tree_index, 6000)
            primary_texts.append(f"### {doc.filename}\n{text}")

        # Load context documents
        context_texts = []
        for doc_id in context_doc_ids:
            doc = await _get_document(db, doc_id)
            if doc and doc.tree_index:
                text = _extract_text(doc.tree_index, 4000)
                context_texts.append(f"### {doc.filename}\n{text}")

        # Load reference document
        reference_text = ""
        if reference_doc_id:
            ref_doc = await _get_document(db, reference_doc_id)
            if ref_doc and ref_doc.tree_index:
                reference_text = _extract_text(ref_doc.tree_index, 4000)

        # Build the composition prompt
        prompt_parts = [
            f"You are a professional document specialist. Your task: {task}\n",
            "=" * 60,
            "\n## PRIMARY DOCUMENT(S) — the main content to work with:\n",
            "\n\n".join(primary_texts),
        ]

        if context_texts:
            prompt_parts.extend([
                "\n\n" + "=" * 60,
                "\n## SUPPORTING CONTEXT — use for background, evidence, and reference:\n",
                "\n\n".join(context_texts),
            ])

        if reference_text:
            prompt_parts.extend([
                "\n\n" + "=" * 60,
                "\n## OUTPUT FORMAT REFERENCE — follow this structure and style:\n",
                reference_text,
            ])

        if instructions:
            prompt_parts.extend([
                "\n\n" + "=" * 60,
                f"\n## ADDITIONAL INSTRUCTIONS:\n{instructions}",
            ])

        prompt_parts.append(
            "\n\n" + "=" * 60
            + "\nNow produce the requested output. Be thorough, professional, "
            "and follow the format reference if provided."
        )

        full_prompt = "\n".join(prompt_parts)
        # Cap total context
        if len(full_prompt) > 30000:
            full_prompt = full_prompt[:30000] + "\n\n[... context truncated for length ...]"

        response = await llm.acomplete(full_prompt)
        content = response.content

        # Truncate for chat context
        if len(content) > 8000:
            preview = content[:8000] + "\n\n... [response truncated — full content is " + str(len(content)) + " characters]"
        else:
            preview = content

        return {
            "task": task,
            "primary_documents": len(primary_doc_ids),
            "context_documents": len(context_doc_ids),
            "has_reference": bool(reference_doc_id),
            "composed_content": preview,
            "full_length": len(content),
        }

    except Exception as exc:
        logger.error("compose_with_context failed: %s", exc)
        return {"error": f"Composition failed: {exc}"}


# ======================================================================
# Dispatcher
# ======================================================================

_EXECUTORS: dict[str, Any] = {
    "search_document": _execute_search_document,
    "list_documents": _execute_list_documents,
    "summarize_section": _execute_summarize_section,
    "extract_data": _execute_extract_data,
    "query_graph": _execute_query_graph,
    "find_cross_references": _execute_find_cross_references,
    "run_smart_tool": _execute_run_smart_tool,
    "generate_report": _execute_generate_report,
    "run_batch": _execute_run_batch,
    "compose_with_context": _execute_compose_with_context,
}


async def execute_tool(
    name: str,
    args: dict,
    llm: LLMClient,
    db: AsyncSession,
) -> dict:
    """Execute an agent tool by name and return its result dict.

    Raises ``KeyError`` if the tool name is unknown.
    """
    executor = _EXECUTORS.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}
    return await executor(args, llm, db)
