"""
common/types.py

Core Pydantic domain models shared across the entire system.

These are the canonical data structures that flow between layers.
Layers should produce and consume these types rather than raw dicts or DataFrames
wherever structured validation matters.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


# ── Enumerations ──────────────────────────────────────────────────────────────

class MappingConfidence(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    UNMATCHED = "unmatched"


class SeriousnessCategory(str, Enum):
    SERIOUS = "serious"
    POTENTIALLY_SERIOUS = "potentially_serious"
    NON_SERIOUS = "non_serious"
    UNKNOWN = "unknown"


class ArmType(str, Enum):
    TREATMENT = "treatment"
    COMPARATOR = "comparator"
    PLACEBO = "placebo"
    UNKNOWN = "unknown"


class EvidenceGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class TriageStatus(str, Enum):
    HIGH_PRIORITY_REVIEW = "high_priority_review"
    MONITOR = "monitor"
    WATCHLIST = "watchlist"
    DEPRIORITIZED = "deprioritized"
    PENDING = "pending"


class CredibilityClass(str, Enum):
    FDA_LABEL = "fda_label"
    REGULATORY = "regulatory"
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    CASE_REPORT = "case_report"
    UNKNOWN = "unknown"


class OutcomeCode(str, Enum):
    DEATH = "DE"
    HOSPITALIZATION = "HO"
    LIFE_THREATENING = "LT"
    DISABILITY = "DS"
    CONGENITAL_ANOMALY = "CA"
    REQUIRED_INTERVENTION = "RI"
    OTHER_SERIOUS = "OT"


class RoleCode(str, Enum):
    PRIMARY_SUSPECT = "PS"
    SECONDARY_SUSPECT = "SS"
    CONCOMITANT = "C"
    INTERACTING = "I"


# ── Domain entities ───────────────────────────────────────────────────────────

class Drug(BaseModel):
    """Normalized drug / active ingredient."""

    drug_id: str = Field(..., description="Internal unique identifier, e.g. 'semaglutide'")
    normalized_name: str
    raw_names_seen: list[str] = Field(default_factory=list)
    brand_names: list[str] = Field(default_factory=list)
    drug_class: str | None = None
    mapping_source: str | None = None
    mapping_confidence: MappingConfidence = MappingConfidence.UNMATCHED


class AdverseEvent(BaseModel):
    """Normalized adverse event / event family."""

    event_id: str = Field(..., description="Internal unique identifier, e.g. 'pancreatitis'")
    normalized_term: str
    event_family: str
    raw_terms_seen: list[str] = Field(default_factory=list)
    seriousness_category: SeriousnessCategory = SeriousnessCategory.UNKNOWN
    system_organ_class: str | None = None
    mapping_source: str | None = None
    mapping_confidence: MappingConfidence = MappingConfidence.UNMATCHED


class ClinicalTrial(BaseModel):
    """One ClinicalTrials.gov study."""

    nct_id: str
    title: str
    sponsor: str | None = None
    phase: list[str] = Field(default_factory=list)
    status: str | None = None
    conditions: list[str] = Field(default_factory=list)
    enrollment: int | None = None
    start_date: date | None = None
    completion_date: date | None = None
    result_available: bool = False
    eligibility_summary: str | None = None
    population_notes: str | None = None


class TrialArm(BaseModel):
    """One arm/group within a clinical trial."""

    arm_id: str
    nct_id: str
    arm_label: str
    arm_type: ArmType = ArmType.UNKNOWN
    intervention: str | None = None
    participant_count: int | None = None
    is_comparator: bool = False


class TrialAdverseEvent(BaseModel):
    """Adverse event reported for one trial arm."""

    nct_id: str
    arm_id: str
    arm_type: ArmType = ArmType.UNKNOWN
    event_term: str
    event_id: str | None = None           # normalized event ID, if mapped
    affected_participants: int | None = None
    at_risk_participants: int | None = None
    event_rate: float | None = None        # affected / at_risk
    is_serious: bool = False


class SignalMetrics(BaseModel):
    """Disproportionality metrics for one drug-event pair."""

    drug_id: str
    event_id: str
    evidence_window: str                   # e.g. "2026Q1"
    case_count: int = 0
    ror: float | None = None
    ror_lower_ci: float | None = None
    ror_upper_ci: float | None = None
    prr: float | None = None
    chi2_p_value: float | None = None
    seriousness_rate: float | None = None  # fraction of cases with serious outcome
    death_count: int = 0
    hospitalization_count: int = 0
    trend_slope: float | None = None
    first_observed_quarter: str | None = None
    potential_publicity_spike: bool = False
    drug_mapping_confidence: MappingConfidence = MappingConfidence.UNMATCHED
    event_mapping_confidence: MappingConfidence = MappingConfidence.UNMATCHED


class EvidenceDocument(BaseModel):
    """One retrieved evidence document from any external source."""

    doc_id: str                            # unique within signal, e.g. "pubmed_12345678"
    source_type: str                       # "pubmed", "fda_safety_comm", "fda_label", etc.
    title: str
    url: str | None = None
    identifier: str | None = None         # DOI, PMID, NDA number, etc.
    publication_date: date | None = None
    retrieved_date: datetime
    snippet: str | None = None            # abstract or relevant extract
    relevance_score: float | None = None  # 0–1
    credibility: CredibilityClass = CredibilityClass.UNKNOWN
    drug_id: str | None = None
    event_id: str | None = None
    accepted: bool = True                 # False = rejected by quality gate
    rejection_reason: str | None = None


class EvidencePacket(BaseModel):
    """
    Complete evidence bundle for one drug-event signal.
    This is the final structured output of the LangGraph workflow.
    """

    signal_id: str                         # "{drug_id}__{event_id}"
    drug_id: str
    event_id: str
    drug_label: str
    event_label: str
    evidence_window: str

    # Quantitative signal evidence
    signal_metrics: SignalMetrics | None = None

    # Clinical trial evidence
    trial_evidence_available: bool = False
    matching_trials: list[ClinicalTrial] = Field(default_factory=list)
    trial_ae_rates: list[TrialAdverseEvent] = Field(default_factory=list)

    # External evidence
    regulatory_documents: list[EvidenceDocument] = Field(default_factory=list)
    literature_documents: list[EvidenceDocument] = Field(default_factory=list)
    label_documents: list[EvidenceDocument] = Field(default_factory=list)
    rejected_evidence: list[EvidenceDocument] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)

    # Synthesis outputs
    synthesis_summary: str | None = None
    synthesis_supports: list[str] = Field(default_factory=list)
    synthesis_contradictions: list[str] = Field(default_factory=list)
    limitation_statement: str | None = None

    # Grading and triage
    evidence_grade: EvidenceGrade | None = None
    grade_explanation: str | None = None
    triage_status: TriageStatus = TriageStatus.PENDING

    # Workflow metadata
    human_review_required: bool = False
    human_review_notes: str | None = None
    report_generated: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def all_accepted_documents(self) -> list[EvidenceDocument]:
        return [
            d for d in
            self.regulatory_documents + self.literature_documents + self.label_documents
            if d.accepted
        ]
