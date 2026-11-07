"""Evaluation framework for retrieval and generation benchmarks."""

from evaluation.metrics import (
    calculate_hit_rate,
    calculate_recall,
    calculate_mrr,
    calculate_ndcg,
    evaluate_retrieval_query,
    evaluate_generation_faithfulness,
)
from evaluation.evaluate_rag import run_evaluation

__all__ = [
    "calculate_hit_rate",
    "calculate_recall",
    "calculate_mrr",
    "calculate_ndcg",
    "evaluate_retrieval_query",
    "evaluate_generation_faithfulness",
    "run_evaluation",
]

