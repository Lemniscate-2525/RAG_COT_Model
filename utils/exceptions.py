"""Domain-specific exceptions for RAG_COT_Model."""

class RAGException(Exception):
    """Base exception for all RAG application errors."""
    pass

class ConfigurationError(RAGException):
    """Raised when application configuration or credentials are missing/invalid."""
    pass

class SSRFSecurityError(RAGException):
    """Raised when a requested URL fails SSRF validation or attempts forbidden access."""
    pass

class IngestionError(RAGException):
    """Raised when fetching external articles fails or times out."""
    pass

class PreprocessingError(RAGException):
    """Raised when parsing or cleaning HTML fails."""
    pass

class ChunkingError(RAGException):
    """Raised when text chunking parameters are invalid or execution fails."""
    pass

class EmbeddingError(RAGException):
    """Raised when vector embedding generation fails."""
    pass

class VectorStoreError(RAGException):
    """Raised when FAISS indexing or search fails."""
    pass

class PromptInjectionError(RAGException):
    """Raised when prompt injection patterns are detected in retrieved content or queries."""
    pass

class GenerationError(RAGException):
    """Raised when LLM answer generation fails."""
    pass

class OutputValidationError(RAGException):
    """Raised when LLM output does not conform to the expected schema or cites invalid sources."""
    pass

