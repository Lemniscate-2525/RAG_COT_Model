"""End-to-end integration and smoke test for the RAG pipeline."""

import pytest
from ingestion.wordpress import Article
from retrieval.chunking import TextChunker
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from generation.llm import RAGGenerator

@pytest.fixture(scope="module")
def sample_corpus():
    return [
        Article(
            post_id=1,
            title="Introduction to Vector Search",
            content="Vector search works by encoding documents and queries into high-dimensional embeddings. Cosine similarity calculates the angular distance between unit vectors.",
            url="https://example.com/vector-search",
            date="2026-11-01",
            source="https://example.com",
        ),
        Article(
            post_id=2,
            title="Understanding Retrieval-Augmented Generation",
            content="Retrieval-Augmented Generation (RAG) grounds language models by injecting relevant retrieved factual documents into the context window.",
            url="https://example.com/rag-guide",
            date="2026-11-02",
            source="https://example.com",
        ),
    ]

def test_full_pipeline_end_to_end(sample_corpus):
    # 1. Chunking
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_articles(sample_corpus)
    assert len(chunks) >= 2

    # 2. Embedding
    embedder = EmbeddingManager()
    vectors = embedder.encode([c.text for c in chunks])
    assert vectors.shape[0] == len(chunks)

    # 3. Vector indexing
    vector_store = FAISSVectorStore(dimension=vectors.shape[1])
    vector_store.add_chunks(chunks, vectors)
    assert vector_store.total_vectors == len(chunks)

    # 4. Retrieval
    retriever = Retriever(vector_store=vector_store, embedding_manager=embedder)
    results = retriever.retrieve("How does RAG ground language models?", top_k=2)
    assert len(results) > 0
    # The top retrieved chunk should be from post_id=2 (RAG guide)
    assert results[0]["post_id"] == 2
    assert "Retrieval-Augmented Generation" in results[0]["text"]

    # 5. Generation (keyless fallback mode test)
    generator = RAGGenerator(api_key=None)
    answer_res = generator.generate_answer("How does RAG ground language models?", results)
    assert answer_res.answer != ""
    assert len(answer_res.evidence_steps) > 0

