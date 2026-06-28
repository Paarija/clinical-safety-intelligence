"""
common/exceptions.py

Custom exception hierarchy for the Clinical Safety Intelligence System.
All domain-specific errors inherit from ClinicalSafetyError so callers
can catch at whatever granularity they need.
"""

from __future__ import annotations


class ClinicalSafetyError(Exception):
    """Base exception for all project errors."""


# ── Acquisition layer ─────────────────────────────────────────────────────────

class DataSourceError(ClinicalSafetyError):
    """Raised when a data source cannot be accessed or downloaded."""


class SourceFileNotFoundError(DataSourceError):
    """Raised when an expected raw source file is missing."""



# ── Parsing layer ─────────────────────────────────────────────────────────────

class ParseError(ClinicalSafetyError):
    """Raised when a raw source file cannot be parsed."""


class SchemaValidationError(ParseError):
    """Raised when a parsed table fails schema validation (missing columns, wrong types)."""

    def __init__(self, table_name: str, details: str) -> None:
        self.table_name = table_name
        self.details = details
        super().__init__(f"Schema validation failed for '{table_name}': {details}")


# ── Normalization layer ───────────────────────────────────────────────────────

class NormalizationError(ClinicalSafetyError):
    """Raised when normalization cannot proceed due to configuration or data errors."""


class DrugNormalizationError(NormalizationError):
    """Raised when drug scope config is missing or malformed."""


class EventNormalizationError(NormalizationError):
    """Raised when event scope config is missing or malformed."""


# ── Signal detection layer ────────────────────────────────────────────────────



class ContingencyTableError(ClinicalSafetyError):
    """Raised when a contingency table cannot be built (e.g., missing join columns)."""



class EvidenceRetrievalError(ClinicalSafetyError):
    """Raised when an evidence retriever fails (network error, API error, etc.)."""


# ── LLM / Orchestration layer ─────────────────────────────────────────────────

class LLMProviderError(ClinicalSafetyError):
    """Raised when the configured LLM provider cannot be initialized or called."""

class WorkflowExecutionError(ClinicalSafetyError):
    """Raised when the LangGraph workflow finishes with errors."""

    def __init__(self, signal_id: str, errors: list[str]) -> None:
        self.signal_id = signal_id
        self.errors = errors
        details = "; ".join(errors) if errors else "workflow did not complete"
        super().__init__(f"Workflow failed for {signal_id}: {details}")


class GuardrailViolationError(ClinicalSafetyError):
    """
    Raised when LLM output contains disallowed clinical language that
    cannot be corrected after max retries.
    """

    def __init__(self, matched_phrase: str, output_snippet: str) -> None:
        self.matched_phrase = matched_phrase
        self.output_snippet = output_snippet
        super().__init__(
            f"Guardrail triggered on phrase '{matched_phrase}'. "
            f"Output snippet: {output_snippet[:200]!r}"
        )



# ── Config layer ──────────────────────────────────────────────────────────────

class ConfigError(ClinicalSafetyError):
    """Raised when project configuration is missing, malformed, or contains invalid values."""
