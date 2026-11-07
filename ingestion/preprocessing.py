"""HTML cleaning and textual preprocessing."""

import html
import re
from bs4 import BeautifulSoup
from utils.exceptions import PreprocessingError
from utils.logging_config import logger

# Tags that typically contain non-article noise
EXCLUDED_TAGS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "header",
    "footer",
    "nav",
    "aside",
    "svg",
    "form",
    "button",
]

def clean_html_content(raw_html: str) -> str:
    """
    Strips noise, boilerplate HTML, and scripts while preserving text paragraphs and structure.
    
    Returns clean, normalized plain text.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""

    try:
        soup = BeautifulSoup(raw_html, "html.parser")

        # Strip unneeded structural and script tags
        for tag in soup.find_all(EXCLUDED_TAGS):
            tag.decompose()

        # Add line break before block elements to retain paragraph/heading separation
        for block_tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote"]):
            block_tag.insert_before("\n\n")

        raw_text = soup.get_text(separator=" ")
        unescaped = html.unescape(raw_text)

        # Fix orphaned punctuation caused by separated tags (e.g., 'link .' -> 'link.')
        unescaped = re.sub(r"\s+([.,;:!?])", r"\1", unescaped)

        # Normalize redundant whitespace while preserving paragraph breaks
        # First reduce multiple horizontal spaces to single space
        cleaned_lines = []
        for line in unescaped.split("\n"):
            normalized_line = re.sub(r"[ \t]+", " ", line).strip()
            if normalized_line:
                cleaned_lines.append(normalized_line)

        # Join lines with newlines
        result = "\n\n".join(cleaned_lines)
        return result.strip()
    except Exception as e:
        logger.error(f"HTML preprocessing failed: {e}")
        raise PreprocessingError(f"Failed to clean HTML content: {e}")

def sanitize_text(text: str) -> str:
    """Basic whitespace normalization and entity unescaping for raw text strings."""
    if not text:
        return ""
    unescaped = html.unescape(text)
    return re.sub(r"\s+", " ", unescaped).strip()
