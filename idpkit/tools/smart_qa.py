"""Smart Q&A — Generate question-answer pairs from documents."""

import json
import logging
from typing import Any

from idpkit.core.llm import LLMClient
from idpkit.core.schemas import ToolResult
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SmartQATool(BaseTool):
    """Generates question-answer pairs from document content for training, quizzes, or FAQs."""

    @property
    def name(self) -> str:
        return "smart_qa"

    @property
    def display_name(self) -> str:
        return "Smart Q&A"

    @property
    def description(self) -> str:
        return "Generate question-answer pairs from documents for training, quizzes, or FAQ creation."

    @property
    def options_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to generate Q&A from",
                },
                "count": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Number of Q&A pairs to generate",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "default": "medium",
                    "description": "Difficulty level of the questions",
                },
                "question_type": {
                    "type": "string",
                    "enum": ["factual", "analytical"],
                    "default": "factual",
                    "description": "Type of questions: factual (recall-based) or analytical (reasoning-based)",
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
        count = options.get("count", 5)
        difficulty = options.get("difficulty", "medium")
        question_type = options.get("question_type", "factual")

        difficulty_guidance = {
            "easy": "Generate straightforward questions that test basic recall and understanding of explicitly stated facts.",
            "medium": "Generate questions that require understanding of context and the ability to connect information across sections.",
            "hard": "Generate challenging questions that require deep analysis, inference, synthesis of multiple points, or critical evaluation.",
        }

        type_guidance = {
            "factual": "Focus on factual, recall-based questions with specific, verifiable answers drawn directly from the document.",
            "analytical": "Focus on analytical, reasoning-based questions that require interpretation, comparison, evaluation, or inference.",
        }

        prompt = f"""Generate {count} question-answer pairs based on the following document.

Document: {doc.filename}
Description: {doc.description or 'N/A'}

Document Tree Structure:
{json.dumps(tree_index, indent=2, default=str)}

Difficulty: {difficulty}
{difficulty_guidance.get(difficulty, difficulty_guidance['medium'])}

Question Type: {question_type}
{type_guidance.get(question_type, type_guidance['factual'])}

Return your response as valid JSON with this structure:
{{
    "document": "{doc.filename}",
    "total_questions": {count},
    "difficulty": "{difficulty}",
    "question_type": "{question_type}",
    "qa_pairs": [
        {{
            "id": 1,
            "question": "The question text",
            "answer": "The comprehensive answer",
            "source_section": "Section title where the answer can be found",
            "difficulty": "{difficulty}",
            "category": "topic or category of the question"
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
                data={"raw_qa": response.content},
            )
        except Exception as e:
            logger.error("SmartQA failed for document %s: %s", document_id, e)
            return ToolResult(tool_name=self.name, status="error", error=str(e))
