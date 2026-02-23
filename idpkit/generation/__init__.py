"""IDP Kit Generation — Document generation and templates.

Provides utilities for generating documents in various formats (DOCX,
Markdown) and for managing reusable templates.

Usage::

    from idpkit.generation import generate_docx, generate_markdown

    # Generate a Word document from markdown content
    path = generate_docx(content, "output.docx")

    # Convert a tree index to markdown
    md = generate_markdown(tree_index)
"""

from .docx_generator import generate_docx
from .markdown_generator import generate_markdown
from .templates import analyze_template, get_template, list_templates, save_template

__all__ = [
    "generate_docx",
    "generate_markdown",
    "analyze_template",
    "list_templates",
    "save_template",
    "get_template",
]
