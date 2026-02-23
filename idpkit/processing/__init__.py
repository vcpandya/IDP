"""IDP Kit Processing — Document processing pipelines.

Provides processing capabilities including entity extraction, summarization,
document comparison, format conversion, and a chainable pipeline.

Usage::

    from idpkit.processing import Pipeline, extract_entities, summarize_text

    # Entity extraction
    entities = await extract_entities(text, llm)

    # Summarization
    summary = await summarize_text(text, llm, length="brief")

    # Pipeline
    pipeline = Pipeline("my-pipeline")
    pipeline.add_step("parse", parse_fn)
    pipeline.add_step("extract", extract_fn)
    result = await pipeline.run({"file_path": "doc.pdf"})
"""

from .pipeline import Pipeline
from .extraction import extract_entities
from .summarization import summarize_text, summarize_tree
from .comparison import compare_documents
from .conversion import convert_format

__all__ = [
    "Pipeline",
    "extract_entities",
    "summarize_text",
    "summarize_tree",
    "compare_documents",
    "convert_format",
]
