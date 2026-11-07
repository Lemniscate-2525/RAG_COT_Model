"""OpenAI generation client with structured output validation, citation grounder, and fallback."""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError

from generation.models import GeneratedAnswer, EvidenceStep
from generation.prompts import SYSTEM_PROMPT_V1, format_retrieval_context, construct_user_prompt
from generation.guardrails import (
    detect_prompt_injection,
    validate_citations,
    sanitize_for_display,
)
from utils.exceptions import GenerationError, OutputValidationError
from utils.logging_config import logger

class RAGGenerator:
    """Handles grounded answer generation and evidence extraction using OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        max_query_length: int = 1000,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_query_length = max_query_length

        self.client: Optional[OpenAI] = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> GeneratedAnswer:
        """
        Generates a validated, grounded answer with user-facing evidence steps.
        """
        # Validate query input
        if not question or not question.strip():
            return GeneratedAnswer(
                answer="No question was provided.",
                evidence_steps=[],
                used_sources=[],
                insufficient_evidence=True,
            )

        clean_question = question.strip()
        if len(clean_question) > self.max_query_length:
            clean_question = clean_question[:self.max_query_length]

        # Check for empty retrieved context
        if not retrieved_docs:
            return GeneratedAnswer(
                answer="No relevant source articles were found in the index to answer your question.",
                evidence_steps=[
                    EvidenceStep(
                        explanation="Vector retrieval returned zero passages exceeding the relevance threshold.",
                        source_ids=[],
                    )
                ],
                used_sources=[],
                insufficient_evidence=True,
            )

        # Build delimited context block
        context_block, source_labels = format_retrieval_context(retrieved_docs)

        # Check for prompt injection signatures in query or context
        suspicious_q, reason_q = detect_prompt_injection(clean_question)
        if suspicious_q:
            logger.warning(f"Query flagged by injection guardrail: {reason_q}")

        # Check if OpenAI API is configured
        if not self.client:
            logger.info("OPENAI_API_KEY is not configured; using retrieval-only fallback response.")
            return self._build_retrieval_only_fallback(clean_question, retrieved_docs, source_labels)

        # Format prompt
        user_prompt = construct_user_prompt(clean_question, context_block)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_V1},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            raw_content = raw_content.strip()

            # Strip markdown code fences if model accidentally wrapped json
            if raw_content.startswith("```"):
                raw_content = raw_content.strip("`")
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:].strip()

            # Parse and validate with Pydantic
            parsed = GeneratedAnswer.model_validate_json(raw_content)

            # Validate and filter source citations
            validated_answer = validate_citations(parsed, source_labels)
            return validated_answer

        except OpenAIError as e:
            logger.error(f"OpenAI API invocation error: {e}")
            return GeneratedAnswer(
                answer=f"Answer generation unavailable due to OpenAI API error ({e}). Retrieved source passages are available below.",
                evidence_steps=[
                    EvidenceStep(
                        explanation="API communication failure prevented grounded completion.",
                        source_ids=[],
                    )
                ],
                used_sources=[],
                insufficient_evidence=True,
            )
        except Exception as e:
            logger.error(f"Failed to parse or validate LLM response: {e}")
            return GeneratedAnswer(
                answer="Answer generation failed due to an output validation error. Relevant retrieved passages are listed below.",
                evidence_steps=[
                    EvidenceStep(
                        explanation="The model output could not be validated against the required safety schema.",
                        source_ids=[],
                    )
                ],
                used_sources=[],
                insufficient_evidence=True,
            )

    def _build_retrieval_only_fallback(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        source_labels: List[str],
    ) -> GeneratedAnswer:
        """Fallback when OPENAI_API_KEY is not set."""
        steps = [
            EvidenceStep(
                explanation=f"Query embedded and matched against index using cosine similarity.",
                source_ids=[],
            ),
            EvidenceStep(
                explanation=f"Retrieved {len(docs)} relevant passage(s) from articles.",
                source_ids=source_labels[:len(docs)],
            ),
            EvidenceStep(
                explanation="LLM generation is disabled (no OPENAI_API_KEY). Inspect cited passages directly.",
                source_ids=source_labels[:len(docs)],
            ),
        ]
        return GeneratedAnswer(
            answer="LLM generation is not configured (missing OPENAI_API_KEY). You can inspect the relevant retrieved passages and source articles below.",
            evidence_steps=steps,
            used_sources=source_labels,
            insufficient_evidence=False,
        )

