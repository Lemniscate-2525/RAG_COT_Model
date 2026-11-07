"""Unit tests for embedding generation and L2 normalization."""

import pytest
import numpy as np
from retrieval.embeddings import EmbeddingManager
from utils.exceptions import EmbeddingError

@pytest.fixture(scope="module")
def embedder():
    return EmbeddingManager()

def test_embedder_dimensions_and_type(embedder):
    texts = ["First passage text", "Second unrelated passage"]
    vectors = embedder.encode(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == embedder.dimension

def test_l2_normalization(embedder):
    texts = ["Cosine similarity requires unit vectors."]
    vectors = embedder.encode(texts, normalize=True)
    # L2 norm of normalized vector must equal 1.0 (within float precision)
    norm = np.linalg.norm(vectors[0])
    assert pytest.approx(norm, abs=1e-5) == 1.0

def test_encode_query(embedder):
    query_vector = embedder.encode_query("How does vector indexing work?")
    assert query_vector.shape == (1, embedder.dimension)
    norm = np.linalg.norm(query_vector[0])
    assert pytest.approx(norm, abs=1e-5) == 1.0

def test_empty_query_raises_error(embedder):
    with pytest.raises(EmbeddingError):
        embedder.encode_query("   ")

