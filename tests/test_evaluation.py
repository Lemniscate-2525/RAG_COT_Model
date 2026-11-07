"""Unit tests for retrieval and generation evaluation metrics."""

import math
import pytest
from evaluation.metrics import (
    calculate_hit_rate,
    calculate_recall,
    calculate_mrr,
    calculate_ndcg,
    evaluate_retrieval_query,
    evaluate_generation_faithfulness,
)

def test_retrieval_metrics_exact_hit():
    retrieved_pids = [101, 102, 103]
    relevant_pids = {101}

    assert calculate_hit_rate(retrieved_pids, relevant_pids) == 1.0
    assert calculate_recall(retrieved_pids, relevant_pids) == 1.0
    assert calculate_mrr(retrieved_pids, relevant_pids) == 1.0  # Rank 1 => 1/1 = 1.0
    assert calculate_ndcg(retrieved_pids, relevant_pids, k=3) == 1.0

def test_retrieval_metrics_second_rank():
    retrieved_pids = [999, 101, 103]
    relevant_pids = {101}

    assert calculate_hit_rate(retrieved_pids, relevant_pids) == 1.0
    assert calculate_recall(retrieved_pids, relevant_pids) == 1.0
    assert calculate_mrr(retrieved_pids, relevant_pids) == 0.5  # Rank 2 => 1/2 = 0.5
    # NDCG: DCG = 1 / log2(2 + 1) = 1 / log2(3), IDCG = 1 / log2(1 + 1) = 1.0
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(calculate_ndcg(retrieved_pids, relevant_pids, k=3), abs=1e-4) == expected_ndcg

def test_retrieval_metrics_no_hit():
    retrieved_pids = [999, 998, 997]
    relevant_pids = {101}

    assert calculate_hit_rate(retrieved_pids, relevant_pids) == 0.0
    assert calculate_recall(retrieved_pids, relevant_pids) == 0.0
    assert calculate_mrr(retrieved_pids, relevant_pids) == 0.0
    assert calculate_ndcg(retrieved_pids, relevant_pids, k=3) == 0.0

def test_evaluate_retrieval_query_deduplication():
    # If same post appears multiple times in retrieved chunks, it should be collapsed to rank 1
    chunks = [
        {"post_id": 101, "chunk_id": "101_c0"},
        {"post_id": 101, "chunk_id": "101_c1"},
        {"post_id": 202, "chunk_id": "202_c0"},
    ]
    relevant_pids = {202}
    # 101 is at rank 1, 202 is at rank 2
    metrics = evaluate_retrieval_query(chunks, relevant_pids, k=2)
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5

def test_generation_faithfulness_metrics():
    answer = "Vector search uses unit normalized embeddings."
    context_passages = ["Vector search relies on normalized embeddings for cosine distance."]
    cited_sources = ["S1"]
    available_sources = ["S1", "S2"]

    metrics = evaluate_generation_faithfulness(
        answer=answer,
        context_passages=context_passages,
        cited_source_labels=cited_sources,
        available_source_labels=available_sources,
    )
    assert metrics["citation_precision"] == 1.0
    assert metrics["lexical_grounding_score"] > 0.0
    assert metrics["has_content"] == 1.0

