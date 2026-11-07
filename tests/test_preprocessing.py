"""Unit tests for HTML cleaning and text preprocessing."""

import pytest
from ingestion.preprocessing import clean_html_content, sanitize_text
from utils.exceptions import PreprocessingError

def test_clean_html_removes_scripts_and_styles():
    html_input = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>alert("attack!");</script>
        </head>
        <body>
            <h1>Article Headline</h1>
            <p>First paragraph with <a href="https://example.com">a link</a>.</p>
            <script>console.log("secret");</script>
        </body>
    </html>
    """
    cleaned = clean_html_content(html_input)
    assert "alert" not in cleaned
    assert "body { color: red; }" not in cleaned
    assert "Article Headline" in cleaned
    assert "First paragraph with a link." in cleaned

def test_clean_html_preserves_structure():
    html_input = "<h2>Heading</h2><p>Paragraph 1</p><p>Paragraph 2</p>"
    cleaned = clean_html_content(html_input)
    assert "Heading" in cleaned
    assert "Paragraph 1" in cleaned
    assert "Paragraph 2" in cleaned
    # Ensure paragraphs are separated
    lines = cleaned.split("\n\n")
    assert len(lines) >= 3

def test_clean_html_empty_input():
    assert clean_html_content("") == ""
    assert clean_html_content(None) == ""

def test_sanitize_text_entities():
    text = "TechCrunch &amp; OpenAI &lt;test&gt;   with   spaces"
    sanitized = sanitize_text(text)
    assert sanitized == "TechCrunch & OpenAI <test> with spaces"

