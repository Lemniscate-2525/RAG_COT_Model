"""High-level retriever coordinating embeddings, FAISS query, and deduplication."""

from typing import List, Dict, Any, Optional
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import FAISSVectorStore
from utils.exceptions import VectorStoreError
from utils.logging_config import logger

class Retriever:
    """Coordinates retrieval of relevant passages using normalized vector search."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = 0.0,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        deduplicate_articles: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar chunks for a given query string.
        """
        if not query or not query.strip():
            return []

        if self.vector_store.total_vectors == 0:
            logger.warning("Retrieval attempted on an empty vector store.")
            return []

        # Generate normalized query embedding
        query_vector = self.embedding_manager.encode_query(query)

        # Execute inner-product search
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            deduplicate_posts=deduplicate_articles,
        )

        # Apply similarity threshold
        filtered_results = [
            r for r in raw_results if r.get("similarity_score", 0.0) >= self.similarity_threshold
        ]

        logger.info(f"Retrieved {len(filtered_results)} chunks for query: '{query[:40]}...'")
        return filtered_results

