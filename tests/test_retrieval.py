"""Unit tests for FAISS vector store, retrieval, deduplication, and persistence."""

import os
import tempfile
import pytest
import numpy as np

from retrieval.chunking import TextChunk
from retrieval.vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from utils.exceptions import VectorStoreError

def _make_dummy_chunks_and_vectors(num_items=5, dim=4):
    chunks = []
    # Make normalized random vectors
    rng = np.random.default_rng(42)
    raw_vecs = rng.standard_normal((num_items, dim)).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    vectors = raw_vecs / norms

    for i in range(num_items):
        chunks.append(
            TextChunk(
                chunk_id=f"doc_{i}_c0",
                post_id=100 + (i % 3),  # Intentionally repeat post_id to test deduplication
                text=f"Text content for document {i}",
                title=f"Article {100 + (i % 3)}",
                url=f"https://example.com/post/{100 + (i % 3)}",
                date="2026-11-07",
                source="https://example.com",
                chunk_index=0,
            )
        )
    return chunks, vectors

def test_empty_vector_store():
    store = FAISSVectorStore(dimension=4)
    assert store.total_vectors == 0
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.search(query, top_k=3)
    assert results == []

def test_add_and_search_vector_store():
    dim = 4
    chunks, vectors = _make_dummy_chunks_and_vectors(num_items=5, dim=dim)
    store = FAISSVectorStore(dimension=dim)
    store.add_chunks(chunks, vectors)
    assert store.total_vectors == 5

    # Query with exact first vector -> score should be approx 1.0
    query = vectors[0:1]
    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc_0_c0"
    assert pytest.approx(results[0]["similarity_score"], abs=1e-5) == 1.0

def test_search_deduplication():
    dim = 4
    chunks, vectors = _make_dummy_chunks_and_vectors(num_items=5, dim=dim)
    store = FAISSVectorStore(dimension=dim)
    store.add_chunks(chunks, vectors)

    query = vectors[0:1]
    # Without dedup
    raw_results = store.search(query, top_k=5, deduplicate_posts=False)
    # With dedup
    dedup_results = store.search(query, top_k=5, deduplicate_posts=True)

    unique_posts = set(r["post_id"] for r in dedup_results)
    assert len(unique_posts) == len(dedup_results)

def test_vector_store_persistence():
    dim = 4
    chunks, vectors = _make_dummy_chunks_and_vectors(num_items=4, dim=dim)
    store = FAISSVectorStore(dimension=dim)
    store.add_chunks(chunks, vectors)

    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_path = os.path.join(tmp_dir, "test.index")
        meta_path = os.path.join(tmp_dir, "test.meta.json")

        store.save(idx_path, meta_path)
        assert os.path.exists(idx_path)
        assert os.path.exists(meta_path)

        loaded_store = FAISSVectorStore.load(idx_path, meta_path)
        assert loaded_store.total_vectors == 4
        assert loaded_store.dimension == dim
        assert len(loaded_store.chunks) == 4
        assert loaded_store.chunks[0].chunk_id == chunks[0].chunk_id

def test_persistence_corrupted_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_path = os.path.join(tmp_dir, "bad.index")
        meta_path = os.path.join(tmp_dir, "bad.meta.json")

        # Create non-existent paths
        with pytest.raises(VectorStoreError):
            FAISSVectorStore.load(idx_path, meta_path)

