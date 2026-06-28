from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from clinical_safety.common.exceptions import ConfigError

load_dotenv()

# ── Config dir resolution ─────────────────────────────────────────────────────

def _find_configs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


_CONFIGS_DIR = _find_configs_dir()


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIGS_DIR / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc


# ── Sub-config models ─────────────────────────────────────────────────────────



class FAERSConfig(BaseModel):
    zip_path: str = "data/raw/faers/faers_ascii_2026q1.zip"
    download_url: str = "https://fis.fda.gov/content/Exports/faers_ascii_2026q1.zip"
    auto_download: bool = True
    request_connect_timeout_sec: float = 10.0
    request_read_timeout_sec: float = 60.0
    quarter: str = "2026Q1"
    file_prefixes: dict[str, str] = Field(default_factory=dict)
    encoding: str = "latin-1"
    delimiter: str = "$"


class ClinicalTrialsConfig(BaseModel):
    page_size: int = 100
    max_pages: int = 50
    request_delay_sec: float = 1.0
    request_connect_timeout_sec: float = 10.0
    request_read_timeout_sec: float = 30.0


class FDAConfig(BaseModel):
    safety_comms_url: str = ""
    openfda_label_url: str = "https://api.fda.gov/drug/label.json"
    request_delay_sec: float = 1.0


class PubMedConfig(BaseModel):
    esearch_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    max_results_per_query: int = 20
    years_back: int = 10
    request_delay_sec: float = 0.34
    request_connect_timeout_sec: float = 10.0
    request_read_timeout_sec: float = 30.0


class DataSourcesConfig(BaseModel):
    faers: FAERSConfig = Field(default_factory=FAERSConfig)
    clinicaltrials: ClinicalTrialsConfig = Field(default_factory=ClinicalTrialsConfig)
    fda: FDAConfig = Field(default_factory=FDAConfig)
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)


class DrugEntry(BaseModel):
    id: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    nda_numbers: list[str] = Field(default_factory=list)


class DrugMatchingConfig(BaseModel):
    fuzzy_threshold: int = 85
    case_sensitive: bool = False
    strip_whitespace: bool = True
    strip_dosage_suffix: bool = True


class DrugScopeConfig(BaseModel):
    drug_class: str = ""
    drug_class_label: str = ""
    drugs: list[DrugEntry] = Field(default_factory=list)
    matching: DrugMatchingConfig = Field(default_factory=DrugMatchingConfig)


class EventFamily(BaseModel):
    id: str
    label: str
    seriousness_category: str = "serious"
    system_organ_class: str = ""
    preferred_terms: list[str] = Field(default_factory=list)
    outcome_codes: list[str] = Field(default_factory=list)
    meddra_codes: list[str] = Field(default_factory=list)


class EventMatchingConfig(BaseModel):
    fuzzy_threshold: int = 80
    case_sensitive: bool = False


class EventScopeConfig(BaseModel):
    event_families: list[EventFamily] = Field(default_factory=list)
    matching: EventMatchingConfig = Field(default_factory=EventMatchingConfig)


class SignalDetectionConfig(BaseModel):
    min_case_count: int = 3
    ror_lower_ci_threshold: float = 1.0
    min_mapping_confidence: str = "alias"
    role_code_filter: list[str] = Field(default_factory=lambda: ["PS"])
    sensitivity_role_codes: list[str] = Field(default_factory=lambda: ["PS", "SS"])
    include_pre_dedup_sensitivity: bool = True
    ci_coverage: float = 0.95


class DisproportionalityConfig(BaseModel):
    ror_ci_method: str = "log_normal"
    prr_enabled: bool = True
    chi2_p_threshold: float = 0.05


class TimeTrendConfig(BaseModel):
    min_quarters_for_trend: int = 3
    spike_fold_change: float = 3.0


class SeriousnessConfig(BaseModel):
    serious_outcome_codes: list[str] = Field(default=["DE", "HO", "LT", "DS"])


class SignalThresholdsConfig(BaseModel):
    signal_detection: SignalDetectionConfig = Field(default_factory=SignalDetectionConfig)
    disproportionality: DisproportionalityConfig = Field(default_factory=DisproportionalityConfig)
    time_trends: TimeTrendConfig = Field(default_factory=TimeTrendConfig)
    seriousness: SeriousnessConfig = Field(default_factory=SeriousnessConfig)


class GeminiLLMConfig(BaseModel):
    model: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_output_tokens: int = 4096
    max_retries: int = 3
    retry_delay_sec: float = 2.0


class LLMProviderConfig(BaseModel):
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)


class GuardrailsConfig(BaseModel):
    disallowed_phrases: list[str] = Field(default_factory=list)
    max_correction_retries: int = 2
    fallback_on_failure: bool = True
    fallback_message: str = "[Synthesis unavailable — output did not pass content guardrails. Manual analyst review required.]"


class ModelProvidersConfig(BaseModel):
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)


class AppConfig(BaseModel):
    """Merged configuration for the entire application."""

    data_sources: DataSourcesConfig
    drug_scope: DrugScopeConfig
    event_scope: EventScopeConfig
    signal_thresholds: SignalThresholdsConfig
    model_providers: ModelProvidersConfig

    def validate_runtime(self) -> None:
        """Validate config values that Pydantic type checks cannot prove safe."""
        errors: list[str] = []
        faers = self.data_sources.faers
        ct = self.data_sources.clinicaltrials
        fda = self.data_sources.fda
        pubmed = self.data_sources.pubmed
        signal = self.signal_thresholds.signal_detection
        dispro = self.signal_thresholds.disproportionality
        trend = self.signal_thresholds.time_trends
        llm = self.model_providers.llm.gemini
        guardrails = self.model_providers.guardrails

        if ct.page_size <= 0:
            errors.append("clinicaltrials.page_size must be positive")
        if ct.max_pages <= 0:
            errors.append("clinicaltrials.max_pages must be positive")
        for name, value in {
            "clinicaltrials.request_delay_sec": ct.request_delay_sec,
            "fda.request_delay_sec": fda.request_delay_sec,
            "pubmed.request_delay_sec": pubmed.request_delay_sec,
            "gemini.retry_delay_sec": llm.retry_delay_sec,
        }.items():
            if value < 0:
                errors.append(f"{name} must be non-negative")
        for name, value in {
            "faers.request_connect_timeout_sec": faers.request_connect_timeout_sec,
            "faers.request_read_timeout_sec": faers.request_read_timeout_sec,
            "clinicaltrials.request_connect_timeout_sec": ct.request_connect_timeout_sec,
            "clinicaltrials.request_read_timeout_sec": ct.request_read_timeout_sec,
            "pubmed.request_connect_timeout_sec": pubmed.request_connect_timeout_sec,
            "pubmed.request_read_timeout_sec": pubmed.request_read_timeout_sec,
        }.items():
            if value <= 0:
                errors.append(f"{name} must be positive")
        if pubmed.max_results_per_query <= 0:
            errors.append("pubmed.max_results_per_query must be positive")
        if pubmed.years_back <= 0:
            errors.append("pubmed.years_back must be positive")
        if signal.min_case_count < 0:
            errors.append("signal_detection.min_case_count must be non-negative")
        if signal.ror_lower_ci_threshold < 0:
            errors.append("signal_detection.ror_lower_ci_threshold must be non-negative")
        if not signal.role_code_filter:
            errors.append("signal_detection.role_code_filter must not be empty")
        if not signal.sensitivity_role_codes:
            errors.append("signal_detection.sensitivity_role_codes must not be empty")
        if not 0 < signal.ci_coverage < 1:
            errors.append("signal_detection.ci_coverage must be between 0 and 1")
        if not 0 <= dispro.chi2_p_threshold <= 1:
            errors.append("disproportionality.chi2_p_threshold must be between 0 and 1")
        if trend.min_quarters_for_trend <= 0:
            errors.append("time_trends.min_quarters_for_trend must be positive")
        if trend.spike_fold_change <= 0:
            errors.append("time_trends.spike_fold_change must be positive")
        if not 0 <= llm.temperature <= 2:
            errors.append("gemini.temperature must be between 0 and 2")
        if llm.max_output_tokens <= 0:
            errors.append("gemini.max_output_tokens must be positive")
        if llm.max_retries < 0:
            errors.append("gemini.max_retries must be non-negative")
        if guardrails.max_correction_retries < 0:
            errors.append("guardrails.max_correction_retries must be non-negative")
        if errors:
            raise ConfigError("; ".join(errors))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Load and merge application configuration.
    """
    try:
        data_sources_raw = _load_yaml("data_sources.yaml")
        drug_scope_raw = _load_yaml("drug_scope.yaml")
        event_scope_raw = _load_yaml("event_scope.yaml")
        signal_thresholds_raw = _load_yaml("signal_thresholds.yaml")
        model_providers_raw = _load_yaml("model_providers.yaml")

        cfg = AppConfig(
            data_sources=DataSourcesConfig.model_validate(data_sources_raw),
            drug_scope=DrugScopeConfig.model_validate(drug_scope_raw),
            event_scope=EventScopeConfig.model_validate(event_scope_raw),
            signal_thresholds=SignalThresholdsConfig.model_validate(signal_thresholds_raw),
            model_providers=ModelProvidersConfig.model_validate(model_providers_raw),
        )
        cfg.validate_runtime()
        return cfg
    except Exception as exc:
        raise ConfigError(f"Failed to load configuration: {exc}") from exc


