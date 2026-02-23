"""Smart Classify — Automatic document categorization and labeling."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartClassifyTool(BaseTool):
    """Auto-categorizes documents by type, topics, sentiment, and urgency."""

    @property
    def name(self) -> str:
        return "smart_classify"

    @property
    def display_name(self) -> str:
        return "Smart Classify"

    @property
    def description(self) -> str:
        return "Classify documents by type, topics, sentiment, and urgency with confidence scores."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to classify",
                },
                "classify_by": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["document_type", "topics", "sentiment", "urgency"],
                    },
                    "default": ["document_type", "topics", "sentiment", "urgency"],
                    "description": "Which classification dimensions to run",
                },
                "custom_labels": {
                    "type": "object",
                    "description": "Custom label sets per dimension, e.g. {'document_type': ['invoice', 'contract', 'memo']}",
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
        classify_by = options.get("classify_by", ["document_type", "topics", "sentiment", "urgency"])
        custom_labels = options.get("custom_labels", {})

        custom_labels_text = ""
        if custom_labels:
            parts = []
            for dim, labels in custom_labels.items():
                parts.append(f"  - {dim}: choose from {json.dumps(labels)}")
            custom_labels_text = "\nCustom label constraints:\n" + "\n".join(parts)

        prompt = f"""Classify the following document across multiple dimensions in a single analysis.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Classification dimensions to evaluate: {json.dumps(classify_by)}
{custom_labels_text}

Return your response as valid JSON with this exact structure:
{{
    "document_type": {{
        "label": "primary document type",
        "confidence": 0.95,
        "secondary_types": ["other possible type"]
    }},
    "topics": {{
        "primary_topics": [
            {{"label": "topic name", "confidence": 0.9}}
        ],
        "tags": ["tag1", "tag2"]
    }},
    "sentiment": {{
        "overall": "positive|negative|neutral|mixed",
        "confidence": 0.85,
        "tone": "formal|informal|technical|persuasive|neutral"
    }},
    "urgency": {{
        "level": "critical|high|medium|low|none",
        "confidence": 0.8,
        "reasoning": "brief explanation"
    }}
}}

Only include dimensions that were requested. Return ONLY the JSON, no other text."""

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
                data={"raw_classification": response.content},
            )
        except Exception as e:
            logger.error("SmartClassify failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
