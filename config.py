"""Application configuration management with environment and Streamlit secret support."""

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass(frozen=True)
class AppConfig:
    """Immutable application settings and security constraints."""

    # LLM Settings
    openai_api_key: Optional[str]
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 1500

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Ingestion & Preprocessing
    default_wordpress_url: str = "https://techcrunch.com"
    max_articles_to_fetch: int = 100
    request_timeout_seconds: int = 20
    max_url_length: int = 2048
    max_response_bytes: int = 15 * 1024 * 1024  # 15 MB max response size

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    default_top_k: int = 3
    max_top_k: int = 15
    similarity_threshold: float = 0.0  # Cosine threshold

    # Security & Guardrails
    max_query_length: int = 1000
    max_context_chars: int = 14000
    allowed_schemes: tuple = ("http", "https")
    allowed_ports: tuple = (80, 443, 8080, 8443)

    # Logging
    log_level: str = "INFO"

def get_config() -> AppConfig:
    """Loads configuration prioritizing Streamlit secrets, then environment variables, then defaults."""
    api_key: Optional[str] = None

    # Check Streamlit secrets first if running in Streamlit runtime
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    return AppConfig(
        openai_api_key=api_key if api_key and api_key.strip() else None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        default_wordpress_url=os.getenv("DEFAULT_WORDPRESS_URL", "https://techcrunch.com"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "3")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

