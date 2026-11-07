"""Pydantic models for structured generation output, evidence steps, and citations."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class EvidenceStep(BaseModel):
    """A distinct factual step linking a claim to supporting source identifiers."""
    explanation: str = Field(..., max_length=1000, description="Concise explanation grounded in the evidence.")
    source_ids: List[str] = Field(default_factory=list, description="List of source labels such as S1, S2.")

    @field_validator("source_ids")
    @classmethod
    def clean_source_ids(cls, v: List[str]) -> List[str]:
        return [s.strip().upper() for s in v if isinstance(s, str) and s.strip()]

class GeneratedAnswer(BaseModel):
    """Structured response schema returned by the grounded RAG generator."""
    answer: str = Field(..., max_length=5000, description="The grounded answer to the user's question.")
    evidence_steps: List[EvidenceStep] = Field(
        default_factory=list,
        max_length=10,
        description="User-facing evidence steps explaining how the answer is supported.",
    )
    used_sources: List[str] = Field(
        default_factory=list,
        description="Source labels (e.g., S1, S2) actually cited in the response.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="Flag set to True if retrieved context lacked sufficient information.",
    )

    @field_validator("used_sources")
    @classmethod
    def clean_used_sources(cls, v: List[str]) -> List[str]:
        return [s.strip().upper() for s in v if isinstance(s, str) and s.strip()]

