"""
Unit tests for deterministic evidence grading.
"""

from __future__ import annotations

from datetime import datetime

from clinical_safety.common.types import (
    CredibilityClass,
    EvidenceDocument,
    EvidenceGrade,
    EvidencePacket,
    MappingConfidence,
    SignalMetrics,
    TriageStatus,
)
from clinical_safety.orchestration.grader import grade_evidence, requires_human_review


FIXED_RETRIEVED_AT = datetime(2024, 1, 1, 12, 0, 0)


def make_support_doc(credibility: CredibilityClass = CredibilityClass.REGULATORY) -> EvidenceDocument:
    return EvidenceDocument(
        doc_id="doc-1",
        source_type="fda_label",
        title="Supporting evidence",
        retrieved_date=FIXED_RETRIEVED_AT,
        credibility=credibility,
    )


def make_packet(
    *,
    drug_conf: MappingConfidence = MappingConfidence.EXACT,
    event_conf: MappingConfidence = MappingConfidence.EXACT,
    case_count: int = 10,
    ror_lower_ci: float = 2.0,
    death_count: int = 1,
    hospitalization_count: int = 0,
    contradictions: list[str] | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        signal_id="drug__event",
        drug_id="drug",
        event_id="event",
        drug_label="Drug",
        event_label="Event",
        evidence_window="2024Q1",
        signal_metrics=SignalMetrics(
            drug_id="drug",
            event_id="event",
            evidence_window="2024Q1",
            case_count=case_count,
            ror_lower_ci=ror_lower_ci,
            death_count=death_count,
            hospitalization_count=hospitalization_count,
            drug_mapping_confidence=drug_conf,
            event_mapping_confidence=event_conf,
        ),
        regulatory_documents=[make_support_doc()],
        synthesis_contradictions=contradictions or [],
    )


def test_a_grade_packet_stays_a_and_updates_side_effects():
    packet = make_packet()

    grade, explanation = grade_evidence(packet)

    assert grade is EvidenceGrade.A
    assert packet.evidence_grade is EvidenceGrade.A
    assert packet.triage_status is TriageStatus.HIGH_PRIORITY_REVIEW
    assert "Base grade: A" in explanation


def test_fuzzy_mapping_caps_a_to_b():
    packet = make_packet(drug_conf=MappingConfidence.FUZZY)

    grade, _ = grade_evidence(packet)

    assert grade is EvidenceGrade.B
    assert packet.evidence_grade is EvidenceGrade.B
    assert packet.triage_status is TriageStatus.MONITOR


def test_unmatched_mapping_forces_d():
    packet = make_packet(event_conf=MappingConfidence.UNMATCHED)

    grade, _ = grade_evidence(packet)

    assert grade is EvidenceGrade.D
    assert packet.evidence_grade is EvidenceGrade.D
    assert packet.triage_status is TriageStatus.DEPRIORITIZED


def test_requires_human_review_handles_grade_a_contradictions_and_clean_lower_grades():
    a_packet = make_packet()
    grade_evidence(a_packet)
    assert requires_human_review(a_packet)

    contradiction_packet = EvidencePacket(
        signal_id="drug__event",
        drug_id="drug",
        event_id="event",
        drug_label="Drug",
        event_label="Event",
        evidence_window="2024Q1",
        evidence_grade=EvidenceGrade.B,
        signal_metrics=SignalMetrics(
            drug_id="drug",
            event_id="event",
            evidence_window="2024Q1",
            case_count=5,
            ror_lower_ci=1.0,
            drug_mapping_confidence=MappingConfidence.EXACT,
            event_mapping_confidence=MappingConfidence.ALIAS,
        ),
        synthesis_contradictions=["trial evidence conflicts with FAERS"],
    )
    assert requires_human_review(contradiction_packet)

    clean_lower_grade_packet = make_packet(
        drug_conf=MappingConfidence.EXACT,
        event_conf=MappingConfidence.ALIAS,
        case_count=5,
        ror_lower_ci=1.0,
        death_count=0,
    )
    grade_evidence(clean_lower_grade_packet)
    assert clean_lower_grade_packet.evidence_grade is EvidenceGrade.B
    assert not requires_human_review(clean_lower_grade_packet)
