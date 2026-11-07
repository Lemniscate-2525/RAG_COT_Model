"""Unit tests for generation models, citation validation, and fallbacks."""

import pytest
from generation.models import GeneratedAnswer, EvidenceStep
from generation.guardrails import validate_citations, detect_prompt_injection, sanitize_for_display
from generation.llm import RAGGenerator

def test_pydantic_schema_validation():
    data = {
        "answer": "This is a factual answer.",
        "evidence_steps": [
            {"explanation": "Fact verified by article.", "source_ids": ["s1", "S2"]}
        ],
        "used_sources": ["s1"],
        "insufficient_evidence": False,
    }
    answer_obj = GeneratedAnswer.model_validate(data)
    assert answer_obj.answer == "This is a factual answer."
    assert answer_obj.evidence_steps[0].source_ids == ["S1", "S2"]
    assert answer_obj.used_sources == ["S1"]

def test_citation_grounder_filters_hallucinated_sources():
    raw_answer = GeneratedAnswer(
        answer="Grounded answer.",
        evidence_steps=[
            EvidenceStep(explanation="Claim from S1", source_ids=["S1"]),
            EvidenceStep(explanation="Claim from fake source", source_ids=["S99"]),
        ],
        used_sources=["S1", "S99", "FABRICATED_URL"],
        insufficient_evidence=False,
    )
    valid_source_ids = ["S1", "S2"]
    sanitized = validate_citations(raw_answer, valid_source_ids)

    # S99 and FABRICATED_URL must be stripped
    assert "S99" not in sanitized.used_sources
    assert "FABRICATED_URL" not in sanitized.used_sources
    assert sanitized.evidence_steps[1].source_ids == []
    assert "S1" in sanitized.used_sources

def test_keyless_generator_fallback():
    # Generator without API key returns structured fallback
    generator = RAGGenerator(api_key=None)
    docs = [
        {"text": "Sample passage content", "title": "Sample Title", "url": "https://example.com"}
    ]
    res = generator.generate_answer("What is in this passage?", docs)
    assert isinstance(res, GeneratedAnswer)
    assert "LLM generation is not configured" in res.answer
    assert len(res.evidence_steps) > 0

def test_empty_retrieved_docs_fallback():
    generator = RAGGenerator(api_key="sk-dummy-key")
    res = generator.generate_answer("Any question?", [])
    assert res.insufficient_evidence is True
    assert "No relevant source articles" in res.answer

