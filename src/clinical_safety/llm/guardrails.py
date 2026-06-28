"""
llm/guardrails.py

Post-generation content guardrail for LLM outputs in the
Clinical Safety Intelligence System.

Responsibility:
  - Scan LLM output text for disallowed phrases (from model_providers.yaml)
  - If found: retry with a correction prompt (up to max_correction_retries)
  - If still failing: return the configured fallback message

This is a deterministic, regex-free guardrail — it uses simple substring
matching so its behaviour is fully auditable.

IMPORTANT: The guardrail does NOT validate clinical facts.  It only blocks
specific prohibited linguistic patterns (causal language, clinical
recommendations, etc.) defined in guardrails.disallowed_phrases.

Usage:
    from clinical_safety.llm.guardrails import Guardrails
    guard = Guardrails(provider)
    safe_text = guard.check_and_correct(raw_text, context_messages)
"""

from __future__ import annotations

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import GuardrailViolationError
from clinical_safety.common.logging import get_logger

logger = get_logger(__name__)

_CORRECTION_PROMPT = (
    "Your previous response contained language that is not appropriate for a "
    "pharmacovigilance signal triage document. Specifically, it contained: '{phrase}'. "
    "\n\nPlease rewrite your response, removing any causal claims, clinical "
    "recommendations, or prescribing guidance. Use hedged language such as "
    "'is associated with', 'reports suggest', 'further investigation is needed'. "
    "Avoid conclusions about discontinuation, causality, diagnosis, or treatment."
)


class Guardrails:
    """
    Post-generation guardrail checker with retry-and-correct logic.

    Args:
        provider: A GeminiProvider (or any object with an invoke() method).
    """

    def __init__(self, provider: object) -> None:
        cfg = get_config()
        gr_cfg = cfg.model_providers.guardrails
        self._disallowed: list[str] = [p.lower() for p in gr_cfg.disallowed_phrases]
        self._max_retries: int = gr_cfg.max_correction_retries
        self._fallback_on_failure: bool = gr_cfg.fallback_on_failure
        self._fallback_message: str = gr_cfg.fallback_message
        self._provider = provider

    def check_and_correct(
        self,
        text: str,
        original_messages: list[dict[str, str]],
    ) -> str:
        """
        Validate text and attempt correction if disallowed phrases are found.

        Args:
            text             : The raw LLM output to validate.
            original_messages: The messages that produced the text (for retry context).

        Returns:
            Validated (and possibly corrected) text.

        Raises:
            GuardrailViolationError: If violations persist after max retries
                                     AND fallback_on_failure is False.
        """
        matched = self._find_violation(text)
        if matched is None:
            return text  # Clean — return as-is

        logger.warning(
            "Guardrail triggered: disallowed phrase '%s' found in LLM output.", matched
        )

        # Attempt correction retries
        current_text = text
        for attempt in range(1, self._max_retries + 1):
            correction_msg = _CORRECTION_PROMPT.format(phrase=matched)
            retry_messages = original_messages + [
                {"role": "ai", "content": current_text},
                {"role": "human", "content": correction_msg},
            ]
            try:
                current_text = self._provider.invoke(retry_messages)  # type: ignore[attr-defined]
                re_matched = self._find_violation(current_text)
                if re_matched is None:
                    logger.info(
                        "Guardrail correction succeeded on attempt %d.", attempt
                    )
                    return current_text
                logger.warning(
                    "Guardrail: correction attempt %d still contains '%s'.",
                    attempt, re_matched,
                )
                matched = re_matched
            except Exception as exc:
                logger.error("Guardrail correction attempt %d failed: %s", attempt, exc)

        # All retries exhausted
        if self._fallback_on_failure:
            logger.warning(
                "Guardrail: returning fallback message after %d failed retries.",
                self._max_retries,
            )
            return self._fallback_message

        raise GuardrailViolationError(
            matched_phrase=matched,
            output_snippet=current_text[:500],
        )

    def passes(self, text: str) -> bool:
        """Return True if text passes all guardrails."""
        return self._find_violation(text) is None

    def _find_violation(self, text: str) -> str | None:
        """Return the first matched disallowed phrase, or None if clean."""
        text_lower = text.lower()
        for phrase in self._disallowed:
            if phrase in text_lower:
                return phrase
        return None
