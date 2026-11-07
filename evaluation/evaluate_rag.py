"""CLI runner for evaluating RAG retrieval and generation metrics."""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from ingestion.wordpress import WordPressClient, Article
from retrieval.chunking import TextChunker
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from generation.llm import RAGGenerator
from evaluation.metrics import evaluate_retrieval_query, evaluate_generation_faithfulness
from utils.logging_config import logger

def run_evaluation(
    dataset_path: str,
    site_url: Optional[str] = None,
    articles: Optional[List[Article]] = None,
    pages: int = 2,
    k: int = 3,
    evaluate_llm: bool = False,
) -> Dict[str, Any]:
    """Runs retrieval and generation evaluation on a labelled dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 1. Fetch or load articles
    if articles is None:
        if not site_url:
            raise ValueError("Either site_url or articles must be provided for evaluation.")
        print(f"Fetching articles from {site_url} (pages={pages})...")
        client = WordPressClient(base_url=site_url)
        articles = client.fetch_articles(max_articles=pages * 20, per_page=20)

    print(f"Loaded {len(articles)} articles. Chunking...")
    chunker = TextChunker(chunk_size=900, chunk_overlap=150)
    chunks = chunker.chunk_articles(articles)
    print(f"Created {len(chunks)} text chunks. Generating embeddings...")

    embedder = EmbeddingManager()
    vectors = embedder.encode([c.text for c in chunks])

    vector_store = FAISSVectorStore(dimension=vectors.shape[1])
    vector_store.add_chunks(chunks, vectors)
    retriever = Retriever(vector_store=vector_store, embedding_manager=embedder)

    generator = RAGGenerator() if evaluate_llm else None

    metric_totals = {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    generation_totals = {"citation_precision": 0.0, "lexical_grounding_score": 0.0}
    evaluated_count = 0

    print("\n--- Running Evaluation Queries ---")
    for case in cases:
        question = case["question"]
        relevant_pids = set(map(int, case.get("relevant_post_ids", [])))

        retrieved_docs = retriever.retrieve(question, top_k=k, deduplicate_articles=False)
        metrics = evaluate_retrieval_query(retrieved_docs, relevant_pids, k=k)

        for m_name, val in metrics.items():
            metric_totals[m_name] += val

        print(f"\nQuery: {question}")
        print(f"  Retrieval @ {k}: " + ", ".join(f"{k_metric}={v:.3f}" for k_metric, v in metrics.items()))

        if evaluate_llm and generator:
            gen_res = generator.generate_answer(question, retrieved_docs)
            gen_metrics = evaluate_generation_faithfulness(
                answer=gen_res.answer,
                context_passages=[d["text"] for d in retrieved_docs],
                cited_source_labels=gen_res.used_sources,
                available_source_labels=[f"S{i+1}" for i in range(len(retrieved_docs))],
            )
            for gm_name, gm_val in gen_metrics.items():
                if gm_name in generation_totals:
                    generation_totals[gm_name] += gm_val
            print(f"  Generation Faithfulness: {gen_metrics}")

        evaluated_count += 1

    macro_averages = {m: round(total / evaluated_count, 4) for m, total in metric_totals.items()}
    print(f"\n================ Macro Averages over {evaluated_count} queries (K={k}) ================")
    for m_name, avg_val in macro_averages.items():
        print(f"  {m_name.upper()}@{k}: {avg_val:.4f}")

    if evaluate_llm and evaluated_count > 0:
        print("\nGeneration Metrics:")
        for gm_name, gm_total in generation_totals.items():
            print(f"  {gm_name}: {gm_total / evaluated_count:.4f}")

    print("\nNote: Retrieval metrics measure passage relevance; they do not measure generation hallucination.")
    return macro_averages

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and generation performance.")
    parser.add_argument("--dataset", default="evaluation/dataset/eval_dataset.json", help="Path to evaluation JSON dataset")
    parser.add_argument("--site", default="https://techcrunch.com", help="WordPress site URL")
    parser.add_argument("--pages", type=int, default=2, help="Number of pages to ingest")
    parser.add_argument("--k", type=int, default=3, help="Top-K cutoff for retrieval")
    parser.add_argument("--eval-generation", action="store_true", help="Also evaluate generation faithfulness")

    args = parser.parse_args()
    try:
        run_evaluation(
            dataset_path=args.dataset,
            site_url=args.site,
            pages=args.pages,
            k=args.k,
            evaluate_llm=args.eval_generation,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
