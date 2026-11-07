"""Configurable text chunker preserving paragraph/sentence boundaries and metadata."""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from ingestion.wordpress import Article
from utils.exceptions import ChunkingError

@dataclass
class TextChunk:
    """Represents an embedded text chunk with source provenance."""
    chunk_id: str
    post_id: int
    text: str
    title: str
    url: str
    date: str
    source: str
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TextChunk":
        return cls(**data)

class TextChunker:
    """Splits article text into bounded overlapping chunks respecting boundaries."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        if chunk_size < 20:
            raise ChunkingError("chunk_size must be at least 20 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be non-negative and less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits a single text string into overlapping chunks preferring paragraph, sentence, and word boundaries.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        if len(clean_text) <= self.chunk_size:
            return [clean_text]

        chunks: List[str] = []
        start = 0
        total_len = len(clean_text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            if end < total_len:
                # Search window for boundary (from 60% of chunk_size to end)
                search_start = start + int(self.chunk_size * 0.6)
                window = clean_text[search_start:end]

                # 1. Try paragraph break
                para_pos = window.rfind("\n\n")
                if para_pos != -1:
                    end = search_start + para_pos + 2
                else:
                    # 2. Try sentence boundary (.!?) followed by space
                    sentence_match = list(re.finditer(r"[.!?]\s", window))
                    if sentence_match:
                        end = search_start + sentence_match[-1].end()
                    else:
                        # 3. Try newline
                        newline_pos = window.rfind("\n")
                        if newline_pos != -1:
                            end = search_start + newline_pos + 1
                        else:
                            # 4. Fallback to whitespace
                            space_pos = window.rfind(" ")
                            if space_pos != -1:
                                end = search_start + space_pos + 1

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            if end >= total_len:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def chunk_article(self, article: Article) -> List[TextChunk]:
        """Splits an article into a list of TextChunk objects with provenance metadata."""
        full_text = f"{article.title}\n\n{article.content}".strip()
        segments = self.split_text(full_text)

        chunk_objects: List[TextChunk] = []
        for idx, segment in enumerate(segments):
            chunk_obj = TextChunk(
                chunk_id=f"{article.post_id}_c{idx}",
                post_id=article.post_id,
                text=segment,
                title=article.title,
                url=article.url,
                date=article.date,
                source=article.source,
                chunk_index=idx,
            )
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_articles(self, articles: List[Article]) -> List[TextChunk]:
        """Chunks a batch of articles."""
        all_chunks: List[TextChunk] = []
        for article in articles:
            all_chunks.extend(self.chunk_article(article))
        return all_chunks
