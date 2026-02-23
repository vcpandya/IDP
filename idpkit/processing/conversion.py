"""Format conversion — convert documents between formats."""

import logging
import os

logger = logging.getLogger(__name__)


def convert_format(
    input_path: str,
    output_format: str,
    output_path: str,
) -> str:
    """Convert a document from one format to another.

    Currently supports:

    - Markdown (.md) to DOCX
    - Plain text (.txt) to DOCX

    Parameters
    ----------
    input_path:
        Path to the source file.
    output_format:
        Target format: ``"docx"`` or ``"md"``.
    output_path:
        Destination path for the converted file.

    Returns
    -------
    str
        The absolute path to the generated output file.

    Raises
    ------
    ValueError
        If the conversion is not supported.
    """
    input_ext = os.path.splitext(input_path)[1].lower()
    output_format = output_format.lower().lstrip(".")

    logger.info(
        "Converting %s (%s) -> %s (%s)",
        input_path, input_ext, output_path, output_format,
    )

    if output_format == "docx":
        return _to_docx(input_path, input_ext, output_path)
    elif output_format == "md" or output_format == "markdown":
        return _to_markdown(input_path, input_ext, output_path)
    else:
        raise ValueError(
            f"Unsupported output format: {output_format!r}. "
            f"Supported: docx, md"
        )


def _to_docx(input_path: str, input_ext: str, output_path: str) -> str:
    """Convert text-based formats to DOCX using the docx_generator."""
    from idpkit.generation.docx_generator import generate_docx

    if input_ext in (".md", ".markdown", ".txt", ".text"):
        with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return generate_docx(content, output_path)
    else:
        raise ValueError(
            f"Cannot convert {input_ext!r} to DOCX. "
            f"Supported source formats: .md, .txt"
        )


def _to_markdown(input_path: str, input_ext: str, output_path: str) -> str:
    """Convert formats to Markdown."""
    if input_ext in (".txt", ".text"):
        with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        # Plain text is already valid markdown; just write it.
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        abs_path = os.path.abspath(output_path)
        logger.info("Markdown conversion complete: %s", abs_path)
        return abs_path
    elif input_ext in (".html", ".htm"):
        from idpkit.parsing.html_parser import HTMLParser

        parser = HTMLParser()
        result = parser.parse(input_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(result.text)
        abs_path = os.path.abspath(output_path)
        logger.info("Markdown conversion complete: %s", abs_path)
        return abs_path
    else:
        raise ValueError(
            f"Cannot convert {input_ext!r} to Markdown. "
            f"Supported source formats: .txt, .html"
        )
