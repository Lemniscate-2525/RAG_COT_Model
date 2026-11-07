"""Sentence Transformer embedding generation with L2 normalization and batching."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.exceptions import EmbeddingError
from utils.logging_config import logger

class EmbeddingManager:
    """Manages sentence embedding generation and normalization."""

    _instance: Optional["EmbeddingManager"] = None
    _model_name: Optional[str] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model if not already loaded."""
        if EmbeddingManager._model is None or EmbeddingManager._model_name != self.model_name:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                EmbeddingManager._model = SentenceTransformer(self.model_name)
                EmbeddingManager._model_name = self.model_name
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise EmbeddingError(f"Could not initialize embedding model: {e}")
        self.model = EmbeddingManager._model
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generates float32 embeddings for a list of texts, normalized with L2 norm for cosine search.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Filter empty strings safely
        safe_texts = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]

        try:
            embeddings = self.model.encode(
                safe_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            # Ensure float32 format for FAISS compatibility
            vectors = np.asarray(embeddings, dtype=np.float32)
            return vectors
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to encode texts: {e}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 2D float32 normalized vector."""
        if not query or not query.strip():
            raise EmbeddingError("Cannot encode empty query.")
        return self.encode([query.strip()], batch_size=1, normalize=True)
