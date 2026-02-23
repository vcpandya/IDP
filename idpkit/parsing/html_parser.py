"""HTML parser using BeautifulSoup for text extraction."""

import logging

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """Extract text and metadata from HTML documents.

    Uses ``beautifulsoup4`` to strip scripts and styles while preserving
    the heading structure via markdown-style prefixes.
    """

    def supported_extensions(self) -> list[str]:
        return ["html", "htm"]

    def parse(self, file_path: str) -> ParseResult:
        """Parse an HTML file and return its text content.

        Parameters
        ----------
        file_path:
            Path to the ``.html`` or ``.htm`` file.

        Raises
        ------
        ImportError
            If beautifulsoup4 is not installed.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install it with: pip install beautifulsoup4"
            )

        logger.info("Parsing HTML: %s", file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            raw_html = fh.read()

        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove script, style, and other non-content tags.
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        # --- Metadata ---
        metadata: dict = {}
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()

        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower()
            content = meta.get("content", "")
            if name == "author" and content:
                metadata["author"] = content
            elif name == "description" and content:
                metadata["description"] = content

        # --- Extract text preserving heading structure ---
        parts: list[str] = []
        _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

        body = soup.find("body") or soup
        for element in body.descendants:
            if element.name in _HEADING_TAGS:
                level = int(element.name[1])
                text = element.get_text(separator=" ", strip=True)
                if text:
                    prefix = "#" * (level + 1)
                    parts.append(f"{prefix} {text}")
            elif element.name == "p":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
            elif element.name == "li":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    parts.append(f"- {text}")
            elif element.name == "table":
                table_md = _html_table_to_markdown(element)
                if table_md:
                    parts.append("")
                    parts.append(table_md)
                    parts.append("")

        # Fall back to plain text extraction if structured approach yields nothing.
        if not parts:
            plain = soup.get_text(separator="\n", strip=True)
            if plain:
                parts.append(plain)

        full_text = "\n".join(parts)
        pages = [{"page": 1, "text": full_text}]

        logger.info("HTML parsing complete: %d characters extracted", len(full_text))

        return ParseResult(
            text=full_text,
            pages=pages,
            metadata=metadata,
            page_count=1,
        )


def _html_table_to_markdown(table_tag) -> str:
    """Convert an HTML <table> tag to a markdown table string."""
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text(separator=" ", strip=True).replace("\n", " "))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Determine max column count.
    max_cols = max(len(r) for r in rows)
    lines: list[str] = []
    header = rows[0] + [""] * (max_cols - len(rows[0]))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    for row in rows[1:]:
        padded = row + [""] * (max_cols - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)
