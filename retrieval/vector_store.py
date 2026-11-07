"""FAISS vector store with normalized cosine similarity, metadata pairing, and persistence."""

import json
import os
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import faiss
import numpy as np

from retrieval.chunking import TextChunk
from utils.exceptions import VectorStoreError
from utils.logging_config import logger

class FAISSVectorStore:
    """FAISS IndexFlatIP vector store holding normalized chunk embeddings and metadata."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []
        self._version_id: Optional[str] = None

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add_chunks(self, chunks: List[TextChunk], vectors: np.ndarray) -> None:
        """
        Adds normalized embeddings and their corresponding metadata chunks to the index.
        """
        if len(chunks) == 0:
            return

        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(f"Count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors.")

        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(f"Dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}.")

        # Ensure float32
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)

        self.index.add(vectors_f32)
        self.chunks.extend(chunks)
        self._version_id = self._compute_fingerprint()
        logger.info(f"Added {len(chunks)} vectors to FAISS index. Total vectors: {self.total_vectors}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        deduplicate_posts: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Performs inner-product search over normalized vectors (equivalent to cosine similarity).
        
        Args:
            query_vector: 2D array of shape (1, dimension).
            top_k: Number of results to return.
            deduplicate_posts: If True, returns only the highest-scoring chunk per article post_id.
        """
        if self.total_vectors == 0:
            return []

        if query_vector.shape[1] != self.dimension:
            raise VectorStoreError(f"Query vector dimension {query_vector.shape[1]} != index dimension {self.dimension}.")

        # Clamp k to available vectors
        search_k = min(max(1, top_k * 4 if deduplicate_posts else top_k), self.total_vectors)
        query_f32 = np.ascontiguousarray(query_vector, dtype=np.float32)

        scores, indices = self.index.search(query_f32, search_k)

        results: List[Dict[str, Any]] = []
        seen_posts = set()

        for score, idx in zip(scores[0], indices[0]):
            # Check for FAISS sentinel value (-1) or out-of-bounds index
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[int(idx)]

            if deduplicate_posts:
                if chunk.post_id in seen_posts:
                    continue
                seen_posts.add(chunk.post_id)

            item = chunk.to_dict()
            item["similarity_score"] = float(score)
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def _compute_fingerprint(self) -> str:
        """Computes a checksum from total vectors, dimension, and chunk IDs."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.dimension}_{self.total_vectors}".encode("utf-8"))
        for chunk in self.chunks:
            hasher.update(chunk.chunk_id.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persists the FAISS index and associated chunk metadata with version integrity."""
        if self.total_vectors == 0:
            raise VectorStoreError("Cannot save empty vector store.")

        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)

        faiss.write_index(self.index, index_path)

        payload = {
            "version_id": self._version_id or self._compute_fingerprint(),
            "dimension": self.dimension,
            "total_chunks": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved FAISS index ({self.total_vectors} vectors) to {index_path} and metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSVectorStore":
        """Loads and validates a persisted FAISS index and its metadata."""
        if not os.path.exists(index_path):
            raise VectorStoreError(f"Index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise VectorStoreError(f"Metadata file not found: {metadata_path}")

        # Basic file size safety check (refuse files > 500MB)
        if os.path.getsize(index_path) > 500 * 1024 * 1024:
            raise VectorStoreError("Index file exceeds safety size threshold (500MB).")

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise VectorStoreError(f"Failed to parse index metadata file: {e}")

        dimension = int(payload.get("dimension", 384))
        saved_chunks = [TextChunk.from_dict(d) for d in payload.get("chunks", [])]

        try:
            loaded_index = faiss.read_index(index_path)
        except Exception as e:
            raise VectorStoreError(f"Failed to read FAISS index: {e}")

        if loaded_index.d != dimension:
            raise VectorStoreError(f"Index dimension mismatch: metadata={dimension}, index={loaded_index.d}")

        if loaded_index.ntotal != len(saved_chunks):
            raise VectorStoreError(f"Vector count mismatch: index has {loaded_index.ntotal}, metadata has {len(saved_chunks)}")

        store = cls(dimension=dimension)
        store.index = loaded_index
        store.chunks = saved_chunks
        store._version_id = payload.get("version_id")
        return store

