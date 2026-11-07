"""Evaluation metrics for retrieval quality (post-level) and generation faithfulness."""

import math
from typing import List, Set, Dict, Any

def calculate_hit_rate(retrieved_post_ids: List[int], relevant_post_ids: Set[int]) -> float:
    """Hit Rate@K: 1.0 if at least one relevant post is in the top-k retrieved list, else 0.0."""
    return 1.0 if any(pid in relevant_post_ids for pid in retrieved_post_ids) else 0.0

def calculate_recall(retrieved_post_ids: List[int], relevant_post_ids: Set[int]) -> float:
    """Recall@K: proportion of all known relevant posts retrieved in top-k."""
    if not relevant_post_ids:
        return 0.0
    return len(set(retrieved_post_ids) & relevant_post_ids) / len(relevant_post_ids)

def calculate_mrr(retrieved_post_ids: List[int], relevant_post_ids: Set[int]) -> float:
    """Mean Reciprocal Rank (MRR): 1 / rank of the first relevant post found (1-indexed)."""
    for rank, pid in enumerate(retrieved_post_ids, start=1):
        if pid in relevant_post_ids:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(retrieved_post_ids: List[int], relevant_post_ids: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@K) using binary relevance."""
    if not relevant_post_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(retrieved_post_ids, start=1)
        if pid in relevant_post_ids
    )

    ideal_hits = min(len(relevant_post_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval_query(
    ranked_chunks: List[Dict[str, Any]],
    relevant_post_ids: Set[int],
    k: int = 3,
) -> Dict[str, float]:
    """
    Evaluates a single query's retrieval results, deduplicating multiple chunks from the same post.
    """
    # Deduplicate post_ids preserving highest rank
    ranked_posts: List[int] = []
    seen = set()
    for chunk in ranked_chunks:
        pid = int(chunk["post_id"])
        if pid not in seen:
            ranked_posts.append(pid)
            seen.add(pid)
        if len(ranked_posts) == k:
            break

    return {
        "hit_rate": calculate_hit_rate(ranked_posts, relevant_post_ids),
        "recall": calculate_recall(ranked_posts, relevant_post_ids),
        "mrr": calculate_mrr(ranked_posts, relevant_post_ids),
        "ndcg": calculate_ndcg(ranked_posts, relevant_post_ids, k=k),
    }

def evaluate_generation_faithfulness(
    answer: str,
    context_passages: List[str],
    cited_source_labels: List[str],
    available_source_labels: List[str],
) -> Dict[str, float]:
    """
    Calculates heuristic generation quality metrics:
    - citation_precision: fraction of cited sources that exist in retrieved context.
    - context_grounding_overlap: lexical containment overlap between answer claims and context.
    - claim_presence: whether answer produced content without empty response.
    """
    if not answer or not answer.strip():
        return {
            "citation_precision": 0.0,
            "lexical_grounding_score": 0.0,
            "has_content": 0.0,
        }

    available_set = set(s.upper() for s in available_source_labels)
    cited_set = set(s.upper() for s in cited_source_labels)

    # Citation validity
    if not cited_set:
        citation_precision = 1.0 if not available_set else 0.5
    else:
        valid_citations = cited_set & available_set
        citation_precision = len(valid_citations) / len(cited_set)

    # Word-level overlap heuristic
    answer_words = set(answer.lower().split()) - {"the", "a", "an", "is", "and", "or", "to", "in", "of", "for", "with"}
    context_words = set(" ".join(context_passages).lower().split())

    if not answer_words:
        grounding_score = 0.0
    else:
        grounding_score = len(answer_words & context_words) / len(answer_words)

    return {
        "citation_precision": round(citation_precision, 4),
        "lexical_grounding_score": round(grounding_score, 4),
        "has_content": 1.0,
    }

