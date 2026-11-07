"""Generation layer package providing grounded prompting, guardrails, and LLM orchestration."""

from generation.models import GeneratedAnswer, EvidenceStep
from generation.llm import RAGGenerator
from generation.guardrails import detect_prompt_injection, validate_citations, sanitize_for_display
from generation.prompts import SYSTEM_PROMPT_V1, format_retrieval_context, construct_user_prompt

__all__ = [
    "GeneratedAnswer",
    "EvidenceStep",
    "RAGGenerator",
    "detect_prompt_injection",
    "validate_citations",
    "sanitize_for_display",
    "SYSTEM_PROMPT_V1",
    "format_retrieval_context",
    "construct_user_prompt",
]

