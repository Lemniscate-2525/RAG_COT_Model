"""Unit tests for text chunking and metadata preservation."""

import pytest
from ingestion.wordpress import Article
from retrieval.chunking import TextChunker, TextChunk
from utils.exceptions import ChunkingError

def test_chunker_initialization_validation():
    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=10)  # Too small
    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=500, chunk_overlap=600)  # Overlap >= size

def test_chunker_short_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text = "Short text."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."

def test_chunker_boundary_preference():
    chunker = TextChunker(chunk_size=80, chunk_overlap=20)
    text = (
        "First sentence here. Second sentence starts now. "
        "Third sentence is rather lengthy and continues further into the text."
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    # Check that sentences aren't randomly sliced mid-word if avoidable
    for chunk in chunks:
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")

def test_chunk_article_metadata_preservation():
    article = Article(
        post_id=999,
        title="Test Article Title",
        content="First paragraph.\n\nSecond paragraph content with more text to split.",
        url="https://example.com/test",
        date="2026-11-07",
        source="https://example.com",
    )
    chunker = TextChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_article(article)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.post_id == 999
        assert c.chunk_id == f"999_c{i}"
        assert c.title == "Test Article Title"
        assert c.url == "https://example.com/test"
        assert c.chunk_index == i
