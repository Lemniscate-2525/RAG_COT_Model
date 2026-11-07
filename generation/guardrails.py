"""Security guardrails: prompt injection detection, citation verification, and output sanitization."""

import html
import re
from typing import List, Set, Tuple, Optional
from generation.models import GeneratedAnswer
from utils.logging_config import logger

# Heuristic patterns identifying potential prompt injection in retrieved content or queries
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|system)\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output|leak)\s+(the\s+)?(system\s+prompt|api\s*key|secret)", re.IGNORECASE),
    re.compile(r"(you\s+are\s+now|act\s+as)\s+(an?\s+unrestricted|a\s+different|a\s+malicious)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:\s*override", re.IGNORECASE),
    re.compile(r"(exfiltrate|send)\s+to\s+https?://", re.IGNORECASE),
]

def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans text for common adversarial prompt injection signatures.
    Returns (is_suspicious, reason).
    """
    if not text or not isinstance(text, str):
        return False, None

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_phrase = match.group(0)
            logger.warning(f"Security event: Suspicious injection pattern detected: '{matched_phrase}'")
            return True, f"Detected injection pattern: '{matched_phrase}'"

    return False, None

def validate_citations(
    generated: GeneratedAnswer,
    valid_source_ids: List[str],
) -> GeneratedAnswer:
    """
    Verifies that all source labels cited by the LLM exist in the retrieved context.
    Removes fabricated or hallucinated citations.
    """
    allowed_set: Set[str] = set(s.upper() for s in valid_source_ids)

    # Filter used_sources
    sanitized_used = [s for s in generated.used_sources if s in allowed_set]
    if len(sanitized_used) != len(generated.used_sources):
        hallucinated = set(generated.used_sources) - allowed_set
        logger.warning(f"Validation event: Removed hallucinated source labels: {hallucinated}")

    # Filter evidence_steps
    sanitized_steps = []
    for step in generated.evidence_steps:
        valid_step_sources = [s for s in step.source_ids if s in allowed_set]
        step.source_ids = valid_step_sources
        sanitized_steps.append(step)

    # If any sources were used in steps, ensure they are also in used_sources
    all_step_sources = set()
    for step in sanitized_steps:
        all_step_sources.update(step.source_ids)

    final_used = sorted(list(set(sanitized_used) | all_step_sources))

    return GeneratedAnswer(
        answer=generated.answer,
        evidence_steps=sanitized_steps,
        used_sources=final_used,
        insufficient_evidence=generated.insufficient_evidence,
    )

def sanitize_for_display(text: str) -> str:
    """Escapes HTML entities to prevent stored or reflected XSS when rendering output."""
    if not text:
        return ""
    # Escape HTML to prevent executable script injection in UI
    escaped = html.escape(text)
    return escaped

