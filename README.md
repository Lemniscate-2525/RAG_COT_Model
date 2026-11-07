# WordPress RAG: Chunking, Retrieval & Evidence Trail

A Streamlit RAG app for public WordPress posts. It fetches posts, cleans HTML, splits long content into overlapping chunks, embeds each chunk, retrieves the closest passages, and displays source evidence alongside the answer.

**Reasoning clarification:** this project provides concise, user-facing evidence steps—not a claim to reveal a model's private internal chain-of-thought. The useful target is an answer that can be checked against cited sources, with uncertainty made explicit.

## Pipeline

1. **Ingestion:** fetch public posts using the WordPress REST API with pagination.
2. **Cleaning:** strip rendered HTML and normalize whitespace.
3. **Chunking:** split into approximately 900-character chunks with 150-character overlap so facts near boundaries are less likely to be lost.
4. **Embedding:** encode chunks with `sentence-transformers/all-MiniLM-L6-v2`.
5. **Indexing:** FAISS `IndexFlatIP` over L2-normalized vectors. Inner product is cosine similarity for normalized vectors.
6. **Retrieval:** embed the question and retrieve up to top-k chunks. k is clamped to the number of indexed vectors and invalid FAISS indices (including -1) are skipped.
7. **Answer generation:** set `OPENAI_API_KEY` to generate an answer and short evidence steps from retrieved passages. Without a key, the app clearly shows a retrieval-only fallback instead of pretending an LLM generated an answer.
8. **Inspection:** review retrieved passages, similarity scores, and source links in Streamlit.

## Run

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to enter a site exposing `/wp-json/wp/v2/posts`, then click **Fetch and index**.

### Optional generation

Set `OPENAI_API_KEY` in your environment before starting Streamlit. Optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`). Never commit API keys. The model is prompted to answer from the retrieved context and return an answer, evidence steps, and source labels. These steps are explanatory summaries—not formal proofs—so inspect the citations.

## Evaluation

Retrieval should be evaluated on a manually labelled dataset of questions and relevant WordPress post IDs. Useful metrics include Hit Rate@k, Recall@k, MRR@k, and nDCG@k. These measure retrieval quality, not generation faithfulness or reasoning quality. Evaluate those separately with manually reviewed reference answers and source-grounding checks. Do not report a metric unless it was actually run on a documented dataset.

## Known limitations

- The active FAISS index lives in Streamlit session memory and is rebuilt after indexing; it is not persistent shared storage.
- No incremental refresh, deletion sync, or robust content-hash deduplication is implemented.
- Character-based chunking is a simple baseline, not semantic chunking.
- Similarity scores are not probabilities of correctness.
- The LLM can still hallucinate or misattribute evidence; verify sources.
- The app does not expose hidden chain-of-thought.

## Files

- `app.py` — active Streamlit RAG app.
- `evaluate_rag.py` — labelled retrieval evaluation script.
- `rag-cot-chatbot.py`, `chatbot_api.py` — legacy prototypes retained for reference.

The active application is Python + Streamlit; it does not require PHP or a WordPress plugin.

## Next improvements

Persistent vector storage, incremental ingestion, per-user isolation, hybrid lexical+dense retrieval, reranking, and a manually reviewed answer faithfulness evaluation set.
