"""
llm/gemini_provider.py

Thin wrapper over Google's current GenAI SDK for the Clinical Safety
Intelligence System.

Responsibilities:
  - Load Gemini API key from environment (.env / GEMINI_API_KEY)
  - Apply model / temperature / token config from model_providers.yaml
  - Retry on transient errors using tenacity
  - Support DRY_RUN mode (env: CLINICAL_SAFETY_DRY_RUN=1) which
    returns a safe placeholder without making any API calls

Usage:
    from clinical_safety.llm.gemini_provider import GeminiProvider
    provider = GeminiProvider()
    response = provider.invoke([
        {"role": "system", "content": "You are a pharmacovigilance analyst..."},
        {"role": "human", "content": "Synthesize the following evidence..."},
    ])
    print(response)   # returns str

Note: With DRY_RUN=1, all invoke() calls return a fixed placeholder string.
"""

from __future__ import annotations

import os
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import LLMProviderError
from clinical_safety.common.logging import get_logger

logger = get_logger(__name__)

_DRY_RUN_ENV = "CLINICAL_SAFETY_DRY_RUN"
_DRY_RUN_RESPONSE = """SYNTHESIS_SUMMARY:
Dry-run mode was used, so no Gemini synthesis was requested and this text is not evidence.

SUPPORTING_EVIDENCE:
- Dry-run placeholder only; review the retrieved evidence documents manually.

CONTRADICTIONS_AND_GAPS:
- None identified in dry-run mode.

LIMITATION_STATEMENT:
Dry-run output verifies workflow wiring only and must not be used for safety interpretation.
"""


class GeminiProvider:
    """
    Gemini LLM provider with retry and dry-run support.

    The provider is lazy-initialised — the GenAI client is not
    created until the first invoke() call, so the module can be imported
    without a valid API key (useful in dry-run / test contexts).
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg.model_providers.llm.gemini
        self._dry_run = os.getenv(_DRY_RUN_ENV, "0").strip() in ("1", "true", "yes")
        self._client = None
        self._api_key = ""

        if self._dry_run:
            logger.info(
                "GeminiProvider: DRY_RUN mode active — no API calls will be made."
            )
        else:
            self._api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not self._api_key or self._api_key == "your_gemini_api_key_here":
                raise LLMProviderError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Add a real key to your .env file or set CLINICAL_SAFETY_DRY_RUN=1 "
                    "to run without LLM calls."
                )
            logger.info(
                "GeminiProvider: model=%s, temperature=%.1f, max_tokens=%d",
                self._cfg.model,
                self._cfg.temperature,
                self._cfg.max_output_tokens,
            )

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """
        Call the Gemini model with a list of role/content messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      Role must be 'system', 'human', or 'ai'.

        Returns:
            Model response as a string.

        Raises:
            LLMProviderError: On unrecoverable API failure.
        """
        if self._dry_run:
            return _DRY_RUN_RESPONSE

        return self._invoke_with_retry(messages)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _invoke_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Internal retry-wrapped invoke. Re-raises after max attempts."""
        client = self._get_client()
        try:
            from google.genai import types

            result = client.models.generate_content(
                model=self._cfg.model,
                contents=_to_prompt(messages),
                config=types.GenerateContentConfig(
                    temperature=self._cfg.temperature,
                    max_output_tokens=self._cfg.max_output_tokens,
                ),
            )
            return result.text or ""
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.warning("GeminiProvider: invoke failed — %s. Retrying...", exc)
            raise

    def _get_client(self):
        """Lazy-initialise the Google GenAI client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMProviderError(
                    "google-genai is not installed. Run: pip install google-genai"
                ) from exc

            self._client = genai.Client(api_key=self._api_key)
        return self._client


def _to_prompt(messages: list[dict[str, str]]) -> str:
    """Flatten role/content messages into a single Gemini prompt."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System instructions:\n{content}")
        elif role in {"ai", "assistant"}:
            parts.append(f"Assistant context:\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)
