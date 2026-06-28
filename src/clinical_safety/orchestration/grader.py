"""
orchestration/grader.py

Deterministic evidence grader for the Clinical Safety Intelligence System.

Uses an in-code A/B/C/D rubric.
No LLM is involved — grading is entirely rule-based so it is fully
auditable and reproducible.

Grade A → Strong signal (high FAERS case count, ROR CI ≥ 2.0,
           serious outcomes, ≥1 regulatory/peer-reviewed source)
Grade B → Moderate signal (lower thresholds, some evidence)
Grade C → Weak/uncertain (minimum case count met, contradictions allowed)
Grade D → Insufficient evidence (catch-all)

Additional modifiers:
  - contradiction_downgrade: if trial evidence contradicts FAERS, downgrade 1 level
  - mapping_confidence_cap: fuzzy mapping → max B; unmatched → D

Usage:
    from clinical_safety.orchestration.grader import grade_evidence
    grade, explanation = grade_evidence(packet)
"""

from __future__ import annotations

from clinical_safety.common.logging import get_logger
from clinical_safety.common.types import (
    CredibilityClass,
    EvidenceGrade,
    EvidencePacket,
    MappingConfidence,
    TriageStatus,
)

logger = get_logger(__name__)

# Grade ordered from strongest to weakest for evaluation
_GRADE_ORDER = [EvidenceGrade.A, EvidenceGrade.B, EvidenceGrade.C, EvidenceGrade.D]
_CREDIBILITY_EXTERNAL = {CredibilityClass.FDA_LABEL, CredibilityClass.REGULATORY, CredibilityClass.PEER_REVIEWED}


def grade_evidence(packet: EvidencePacket) -> tuple[EvidenceGrade, str]:
    """
    Apply the deterministic A/B/C/D grading rubric to an EvidencePacket.

    Args:
        packet: A fully assembled EvidencePacket (signals + retrieved evidence).

    Returns:
        (grade, explanation) — explanation is a human-readable string
        describing which criteria were met and which caused downgrades.
    """
    notes: list[str] = []

    # ── Pull signal metrics safely ─────────────────────────────────────────────
    m = packet.signal_metrics
    case_count = m.case_count if m else 0
    ror_lower_ci = m.ror_lower_ci if m else 0.0
    has_serious = (m.death_count > 0 or m.hospitalization_count > 0) if m else False
    drug_conf = m.drug_mapping_confidence if m else MappingConfidence.UNMATCHED
    event_conf = m.event_mapping_confidence if m else MappingConfidence.UNMATCHED

    # ── Count unresolved contradictions ───────────────────────────────────────
    unresolved_contradictions = len(packet.synthesis_contradictions)

    # ── Count external supporting documents ───────────────────────────────────
    external_docs = [
        d for d in packet.all_accepted_documents
        if d.credibility in _CREDIBILITY_EXTERNAL
    ]
    has_external_support = len(external_docs) >= 1

    # ── Try each grade in descending strength order ───────────────────────────
    grade = EvidenceGrade.D
    for candidate_grade in [EvidenceGrade.A, EvidenceGrade.B, EvidenceGrade.C]:
        if _meets_grade(
            candidate_grade,
            case_count,
            ror_lower_ci,
            has_serious,
            has_external_support,
            unresolved_contradictions,
        ):
            grade = candidate_grade
            notes.append(f"Base grade: {candidate_grade.value} (criteria met)")
            break
    else:
        notes.append("Base grade: D (no criteria met — insufficient evidence)")

    # ── Apply mapping confidence cap ──────────────────────────────────────────
    worst_conf = _worst_confidence(drug_conf, event_conf)
    if worst_conf == MappingConfidence.UNMATCHED:
        if grade != EvidenceGrade.D:
            notes.append("Downgraded to D: unmatched drug or event mapping confidence")
            grade = EvidenceGrade.D
    elif worst_conf == MappingConfidence.FUZZY:
        if grade == EvidenceGrade.A:
            notes.append("Grade A capped to B: fuzzy drug or event mapping confidence")
            grade = EvidenceGrade.B

    # ── Apply contradiction downgrade ─────────────────────────────────────────
    if unresolved_contradictions > 0 and grade != EvidenceGrade.D:
        downgraded = _downgrade(grade)
        notes.append(
            f"Downgraded from {grade.value} to {downgraded.value}: "
            f"{unresolved_contradictions} unresolved contradiction(s) found"
        )
        grade = downgraded

    # ── Determine triage status from final grade ───────────────────────────────
    triage = _grade_to_triage(grade)
    packet.evidence_grade = grade
    packet.triage_status = triage

    explanation = " | ".join(notes) if notes else "No criteria evaluated"
    logger.info(
        "Evidence grader: signal=%s → grade=%s, triage=%s | %s",
        packet.signal_id, grade.value, triage.value, explanation,
    )
    return grade, explanation


def _meets_grade(
    grade: EvidenceGrade,
    case_count: int,
    ror_lower_ci: float | None,
    has_serious: bool,
    has_external_support: bool,
    unresolved_contradictions: int,
) -> bool:
    """Check whether the metrics satisfy a specific grade's criteria."""
    ror_ci = ror_lower_ci or 0.0

    if grade == EvidenceGrade.A:
        return (
            case_count >= 10
            and ror_ci >= 2.0
            and has_serious
            and has_external_support
            and unresolved_contradictions == 0
        )
    if grade == EvidenceGrade.B:
        return (
            case_count >= 5
            and ror_ci >= 1.0
            and unresolved_contradictions <= 1
        )
    if grade == EvidenceGrade.C:
        return (
            case_count >= 3
            and ror_ci >= 0.5
        )
    return False  # D is catch-all


def _downgrade(grade: EvidenceGrade) -> EvidenceGrade:
    """Return the next weaker grade."""
    idx = _GRADE_ORDER.index(grade)
    if idx + 1 < len(_GRADE_ORDER):
        return _GRADE_ORDER[idx + 1]
    return EvidenceGrade.D


def _worst_confidence(
    drug_conf: MappingConfidence, event_conf: MappingConfidence
) -> MappingConfidence:
    """Return the worst (least confident) of two MappingConfidence values."""
    _order = [
        MappingConfidence.UNMATCHED,
        MappingConfidence.FUZZY,
        MappingConfidence.ALIAS,
        MappingConfidence.EXACT,
    ]
    drug_idx = _order.index(drug_conf) if drug_conf in _order else 0
    event_idx = _order.index(event_conf) if event_conf in _order else 0
    return _order[min(drug_idx, event_idx)]


def _grade_to_triage(grade: EvidenceGrade) -> TriageStatus:
    """Map evidence grade to triage status."""
    return {
        EvidenceGrade.A: TriageStatus.HIGH_PRIORITY_REVIEW,
        EvidenceGrade.B: TriageStatus.MONITOR,
        EvidenceGrade.C: TriageStatus.WATCHLIST,
        EvidenceGrade.D: TriageStatus.DEPRIORITIZED,
    }.get(grade, TriageStatus.PENDING)


def requires_human_review(packet: EvidencePacket) -> bool:
    """
    Determine whether a signal requires human review checkpoint.

    Rules (from langgraph_workflow.md):
      - Grade A always requires human review
      - Contradiction flagged → human review
      - Low mapping confidence → human review
    """
    if packet.evidence_grade == EvidenceGrade.A:
        return True
    if packet.synthesis_contradictions:
        return True
    m = packet.signal_metrics
    if m and m.drug_mapping_confidence in (MappingConfidence.FUZZY, MappingConfidence.UNMATCHED):
        return True
    return False
