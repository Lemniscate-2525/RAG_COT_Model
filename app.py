"""Streamlit frontend for RAG_COT_Model: Explainable Evidence-Grounded Assistant."""

import os
from typing import List, Dict, Any, Optional
import streamlit as st

from config import get_config
from ingestion.wordpress import WordPressClient, Article
from ingestion.url_validator import validate_target_url
from retrieval.chunking import TextChunker, TextChunk
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from generation.llm import RAGGenerator
from generation.models import GeneratedAnswer
from generation.guardrails import sanitize_for_display
from utils.exceptions import RAGException, SSRFSecurityError
from utils.logging_config import logger

# Configure Streamlit page layout
st.set_page_config(
    page_title="RAG_COT — Explainable Evidence Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load application configuration
config = get_config()

@st.cache_resource(show_spinner=False)
def get_cached_embedding_manager(model_name: str) -> EmbeddingManager:
    """Loads and caches the SentenceTransformer model in memory across user sessions."""
    logger.info(f"Initializing cached EmbeddingManager for: {model_name}")
    return EmbeddingManager(model_name=model_name)

# Initialize user session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "article_count" not in st.session_state:
    st.session_state.article_count = 0

if "source_site" not in st.session_state:
    st.session_state.source_site = config.default_wordpress_url

# Sidebar Navigation & Ingestion Controls
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.markdown("---")

    st.subheader("Content Ingestion")
    site_url = st.text_input(
        "WordPress Site URL",
        value=st.session_state.source_site,
        help="Public WordPress website URL exposing the REST API (/wp-json/wp/v2/posts).",
    )

    col_articles, col_pages = st.columns(2)
    with col_articles:
        num_articles = st.number_input("Max Articles", min_value=5, max_value=config.max_articles_to_fetch, value=25, step=5)
    with col_pages:
        per_page = st.number_input("Per Page", min_value=5, max_value=50, value=20, step=5)

    st.markdown("---")
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K Passages", min_value=1, max_value=config.max_top_k, value=config.default_top_k)
    dedup_posts = st.checkbox("Deduplicate Articles", value=True, help="Return at most one chunk per article in results.")

    st.markdown("---")
    st.subheader("Index Status")
    vs: Optional[FAISSVectorStore] = st.session_state.vector_store
    if vs is not None and vs.total_vectors > 0:
        st.success(f"🟢 Index Active: **{vs.total_vectors}** chunks ({st.session_state.article_count} articles)")
    else:
        st.warning("⚪ Index Not Initialized")

    # Ingestion button
    ingest_clicked = st.button("🚀 Fetch and Index Articles", type="primary", use_container_width=True)

    # Chat controls
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Diagnostics & Security badges
    st.caption("🔒 Security Guardrails: SSRF protection, prompt injection boundary isolation, citation validation.")
    has_key = bool(config.openai_api_key)
    provider_name = "Puter AI" if (config.openai_base_url and "puter" in config.openai_base_url.lower()) else ("Custom Endpoint" if config.openai_base_url else "OpenAI")
    st.caption(f"🧠 Generation: {f'{provider_name} ({config.openai_model})' if has_key else 'Retrieval-Only Mode (No API Key)'}")

# Ingestion Execution Handler
if ingest_clicked:
    try:
        # Validate URL scheme and SSRF safeguards
        validate_target_url(site_url)

        with st.status("Ingesting and Indexing Articles...", expanded=True) as status:
            st.write("1. Validating URL & Connecting to WordPress REST API...")
            client = WordPressClient(base_url=site_url, timeout=config.request_timeout_seconds)
            articles: List[Article] = client.fetch_articles(max_articles=int(num_articles), per_page=int(per_page))
            st.write(f"✓ Ingested {len(articles)} articles.")

            st.write("2. Cleaning HTML & Splitting into Overlapping Chunks...")
            chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            chunks: List[TextChunk] = chunker.chunk_articles(articles)
            st.write(f"✓ Created {len(chunks)} text chunks.")

            st.write("3. Generating Vector Embeddings...")
            embedder = get_cached_embedding_manager(config.embedding_model_name)
            vectors = embedder.encode([c.text for c in chunks], batch_size=config.embedding_batch_size)
            st.write(f"✓ Generated {vectors.shape[0]} embeddings ({vectors.shape[1]}-d).")

            st.write("4. Building FAISS IndexFlatIP...")
            new_vs = FAISSVectorStore(dimension=vectors.shape[1])
            new_vs.add_chunks(chunks, vectors)

            # Store in session state
            st.session_state.vector_store = new_vs
            st.session_state.article_count = len(articles)
            st.session_state.source_site = site_url
            status.update(label=f"Ready! Indexed {len(chunks)} chunks from {len(articles)} articles.", state="complete", expanded=False)

        st.rerun()

    except SSRFSecurityError as e:
        st.error(f"Security Alert (SSRF Blocked): {e}")
        logger.warning(f"User entered blocked URL '{site_url}': {e}")
    except RAGException as e:
        st.error(f"Ingestion Failed: {e}")
        logger.error(f"Ingestion error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during ingestion: {str(e)}")
        logger.exception("Unexpected error during article ingestion")

# Main Interface Header
st.title("🔎 RAG_COT — Explainable Evidence Chatbot")
st.markdown(
    "Ask questions grounded in indexed WordPress articles. Every claim is accompanied by a "
    "**verifiable evidence trail** citing specific source passages."
)

# Display Chat History
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    with st.chat_message(role):
        st.markdown(sanitize_for_display(msg.get("content", "")))

        # If assistant message has evidence steps, display them
        steps = msg.get("evidence_steps", [])
        if steps:
            with st.expander("🔍 Evidence Trail & Reasoning", expanded=False):
                for i, step in enumerate(steps, 1):
                    src_tag = f" `[{', '.join(step.get('source_ids', []))}]`" if step.get("source_ids") else ""
                    st.markdown(f"**Step {i}:** {sanitize_for_display(step.get('explanation', ''))}{src_tag}")

        # If assistant message has source passages, display them
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 Retrieved Source Passages ({len(sources)})", expanded=False):
                for s in sources:
                    score = s.get("similarity_score", 0.0)
                    st.markdown(f"**[{s.get('source_label', 'S')}] [{s.get('title', 'Source')}]({s.get('url', '#')})** — *Cosine Similarity: {score:.3f}*")
                    st.text(s.get("text", "")[:400] + ("..." if len(s.get("text", "")) > 400 else ""))
                    st.divider()

# User Input Box
current_vs: Optional[FAISSVectorStore] = st.session_state.vector_store

if current_vs is None or current_vs.total_vectors == 0:
    st.info("👈 Enter a WordPress URL in the sidebar and click **'Fetch and Index Articles'** to get started.")
else:
    user_query = st.chat_input("Ask a question about the indexed articles...")
    if user_query:
        # Check input bounds
        clean_query = user_query.strip()
        if len(clean_query) > config.max_query_length:
            st.warning(f"Query truncated to maximum allowed {config.max_query_length} characters.")
            clean_query = clean_query[:config.max_query_length]

        # 1. Render user message
        st.session_state.messages.append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(sanitize_for_display(clean_query))

        # 2. Retrieve passages
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and synthesizing grounded answer..."):
                embedder = get_cached_embedding_manager(config.embedding_model_name)
                retriever = Retriever(vector_store=current_vs, embedding_manager=embedder)
                docs = retriever.retrieve(clean_query, top_k=top_k, deduplicate_articles=dedup_posts)

                # Attach source labels S1, S2...
                for idx, d in enumerate(docs, 1):
                    d["source_label"] = f"S{idx}"

                generator = RAGGenerator(
                    api_key=config.openai_api_key,
                    model_name=config.openai_model,
                    temperature=config.openai_temperature,
                )
                generated: GeneratedAnswer = generator.generate_answer(clean_query, docs)

                # Filter docs to those used if used_sources was specified
                used_set = set(generated.used_sources)
                relevant_docs = [d for d in docs if not used_set or d["source_label"] in used_set] if docs else []

                # Render answer
                st.markdown(sanitize_for_display(generated.answer))

                # Render evidence steps
                if generated.evidence_steps:
                    with st.expander("🔍 Evidence Trail & Reasoning", expanded=True):
                        for i, step in enumerate(generated.evidence_steps, 1):
                            src_tag = f" `[{', '.join(step.source_ids)}]`" if step.source_ids else ""
                            st.markdown(f"**Step {i}:** {sanitize_for_display(step.explanation)}{src_tag}")

                # Render sources
                if relevant_docs:
                    with st.expander(f"📚 Retrieved Source Passages ({len(relevant_docs)})", expanded=False):
                        for s in relevant_docs:
                            score = s.get("similarity_score", 0.0)
                            st.markdown(f"**[{s.get('source_label', 'S')}] [{s.get('title', 'Source')}]({s.get('url', '#')})** — *Cosine Similarity: {score:.3f}*")
                            st.text(s.get("text", "")[:400] + ("..." if len(s.get("text", "")) > 400 else ""))
                            st.divider()

                # Save assistant response to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": generated.answer,
                    "evidence_steps": [s.model_dump() for s in generated.evidence_steps],
                    "sources": relevant_docs,
                    "insufficient_evidence": generated.insufficient_evidence,
                })
