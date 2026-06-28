"""Focused integration coverage for the LangGraph evidence workflow."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pytest

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import WorkflowExecutionError
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import (
    CredibilityClass,
    EvidenceDocument,
    MappingConfidence,
    SignalMetrics,
)
from clinical_safety.evidence.retrievers.fda_retriever import FDARetriever
from clinical_safety.evidence.retrievers.pubmed_retriever import PubMedRetriever
from clinical_safety.orchestration.graph import run_signal


FIXED_RETRIEVED_AT = datetime(2024, 1, 1, 12, 0, 0)


def _minimal_signal_metrics(event_id: str = "pancreatitis") -> SignalMetrics:
    return SignalMetrics(
        drug_id="semaglutide",
        event_id=event_id,
        evidence_window="2024Q1",
        case_count=12,
        ror=2.4,
        ror_lower_ci=1.3,
        ror_upper_ci=4.1,
        prr=2.1,
        chi2_p_value=0.01,
        seriousness_rate=0.25,
        death_count=0,
        hospitalization_count=3,
        drug_mapping_confidence=MappingConfidence.EXACT,
        event_mapping_confidence=MappingConfidence.EXACT,
    )


def _normalize_event_terms(event_terms: object) -> list[str]:
    if event_terms is None:
        return []
    if isinstance(event_terms, str):
        return [event_terms]
    if isinstance(event_terms, Iterable):
        return [str(term) for term in event_terms]
    return [str(event_terms)]


def _preferred_terms_for_event(event_id: str) -> tuple[list[str], str]:
    for fam in get_config().event_scope.event_families:
        if fam.id == event_id:
            return fam.preferred_terms or [fam.label], fam.label
    fallback = event_id.replace("_", " ").title()
    return [fallback], fallback


def _patch_retrievers_for_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {"fda": None, "pubmed": None}

    def retrieve_fda(
        self: FDARetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | list[str] | None = None,
        event_terms: list[str] | None = None,
        **kwargs: object,
    ) -> list[EvidenceDocument]:
        # Support old (event_name) and updated (event_terms) calling styles.
        if event_terms is not None:
            captured["fda"] = event_terms
        elif event_name is None:
            captured["fda"] = kwargs.get("event_terms") or kwargs.get("event_name")
        else:
            captured["fda"] = event_name
        return []

    def retrieve_pubmed(
        self: PubMedRetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | list[str] | None = None,
        event_terms: list[str] | None = None,
        **kwargs: object,
    ) -> list[EvidenceDocument]:
        # Support old (event_name) and updated (event_terms) calling styles.
        if event_terms is not None:
            captured["pubmed"] = event_terms
        elif event_name is None:
            captured["pubmed"] = kwargs.get("event_terms") or kwargs.get("event_name")
        else:
            captured["pubmed"] = event_name
        return []

    monkeypatch.setattr(FDARetriever, "retrieve", retrieve_fda)
    monkeypatch.setattr(PubMedRetriever, "retrieve", retrieve_pubmed)
    return captured


def _patch_local_retrievers(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve_fda(
        self: FDARetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | None = None,
        event_terms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        used_event_name = event_name or ", ".join(event_terms or [])
        return [
            EvidenceDocument(
                doc_id="fda-label-local",
                source_type="fda_label",
                title=f"{drug_name} label discusses {used_event_name}",
                url="https://example.test/fda-label",
                retrieved_date=FIXED_RETRIEVED_AT,
                snippet=(
                    f"The local FDA label excerpt mentions {drug_name} and {used_event_name} "
                    "with enough relevant text to pass the quality gate."
                ),
                relevance_score=0.95,
                credibility=CredibilityClass.FDA_LABEL,
                drug_id=drug_id,
                event_id=event_id,
            )
        ]

    def retrieve_pubmed(
        self: PubMedRetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | None = None,
        event_terms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        used_event_name = event_name or ", ".join(event_terms or [])
        return [
            EvidenceDocument(
                doc_id="pubmed-local",
                source_type="pubmed",
                title=f"Case literature for {drug_name} and {used_event_name}",
                url="https://example.test/pubmed",
                identifier="PMID-LOCAL",
                retrieved_date=FIXED_RETRIEVED_AT,
                snippet=(
                    f"A deterministic literature abstract references {drug_name} and {used_event_name} "
                    "so the LangGraph synthesizer path is exercised in dry-run mode."
                ),
                relevance_score=0.9,
                credibility=CredibilityClass.PEER_REVIEWED,
                drug_id=drug_id,
                event_id=event_id,
            )
        ]

    monkeypatch.setattr(FDARetriever, "retrieve", retrieve_fda)
    monkeypatch.setattr(PubMedRetriever, "retrieve", retrieve_pubmed)


@pytest.fixture()
def isolated_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLINICAL_SAFETY_DRY_RUN", "1")
    monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
    _patch_local_retrievers(monkeypatch)
    return Paths(data_dir)


def test_run_signal_dry_run_completes_real_graph_and_generates_report(isolated_workflow: Paths):
    packet = run_signal(
        drug_id="semaglutide",
        event_id="pancreatitis",
        signal_metrics=_minimal_signal_metrics(),
        evidence_window="2024Q1",
        paths=isolated_workflow,
    )

    report_path = isolated_workflow.processed_reports / "semaglutide__pancreatitis_report.md"
    assert packet.report_generated is True
    assert report_path.exists()
    assert packet.synthesis_summary
    assert "Dry-run mode" in packet.synthesis_summary
    assert packet.all_accepted_documents


def test_run_signal_retrieval_uses_preferred_terms_for_composite_event_family(
    monkeypatch: pytest.MonkeyPatch,
    isolated_workflow: Paths,
):
    event_id = "severe_nausea_vomiting"
    preferred_terms, event_label = _preferred_terms_for_event(event_id)
    captured_terms = _patch_retrievers_for_capture(monkeypatch)

    packet = run_signal(
        drug_id="semaglutide",
        event_id=event_id,
        signal_metrics=_minimal_signal_metrics(event_id),
        evidence_window="2024Q1",
        paths=isolated_workflow,
    )

    fda_terms = _normalize_event_terms(captured_terms["fda"])
    pubmed_terms = _normalize_event_terms(captured_terms["pubmed"])
    expected = {term.lower() for term in preferred_terms}

    assert packet.event_label == event_label
    assert expected.issubset({term.lower() for term in fda_terms})
    assert expected.issubset({term.lower() for term in pubmed_terms})
    assert event_label.lower() not in {term.lower() for term in fda_terms}
    assert event_label.lower() not in {term.lower() for term in pubmed_terms}


def test_run_signal_zero_retrieved_documents_does_not_report_all_rejected(
    monkeypatch: pytest.MonkeyPatch,
    isolated_workflow: Paths,
):
    _patch_retrievers_for_capture(monkeypatch)

    packet = run_signal(
        drug_id="semaglutide",
        event_id="severe_nausea_vomiting",
        signal_metrics=_minimal_signal_metrics("severe_nausea_vomiting"),
        evidence_window="2024Q1",
        paths=isolated_workflow,
    )

    assert not packet.all_accepted_documents
    assert not any(
        "All retrieved documents were rejected by the quality gate" in gap
        for gap in packet.evidence_gaps
    )


def test_run_signal_rejects_drug_only_fda_safety_comm_without_event_term(
    monkeypatch: pytest.MonkeyPatch,
    isolated_workflow: Paths,
):
    def retrieve_fda(
        self: FDARetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | None = None,
        event_terms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        return [
            EvidenceDocument(
                doc_id="fda-safety-only-drug",
                source_type="fda_safety_comm",
                title=f"FDA warns about {drug_name} handling",
                url="https://example.test/fda-safety",
                retrieved_date=FIXED_RETRIEVED_AT,
                snippet="General safety communication text with enough length but no matching event family terms.",
                relevance_score=0.7,
                credibility=CredibilityClass.FDA_LABEL,
                drug_id=drug_id,
                event_id=event_id,
            )
        ]

    def retrieve_pubmed(
        self: PubMedRetriever,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str | None = None,
        event_terms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        return []

    monkeypatch.setattr(FDARetriever, "retrieve", retrieve_fda)
    monkeypatch.setattr(PubMedRetriever, "retrieve", retrieve_pubmed)

    packet = run_signal(
        drug_id="semaglutide",
        event_id="severe_nausea_vomiting",
        signal_metrics=_minimal_signal_metrics("severe_nausea_vomiting"),
        evidence_window="2024Q1",
        paths=isolated_workflow,
    )

    assert not packet.all_accepted_documents
    assert packet.rejected_evidence
    assert packet.rejected_evidence[0].source_type == "fda_safety_comm"
    assert packet.rejected_evidence[0].rejection_reason == "FDA safety communication missing event term"

def test_run_signal_raises_when_graph_records_errors_and_does_not_report(
    isolated_workflow: Paths,
):
    analytics_dir = isolated_workflow.processed_analytics
    analytics_dir.mkdir(parents=True)
    (analytics_dir / "trial_comparison.parquet").write_text(
        "not a parquet file", encoding="utf-8"
    )

    with pytest.raises(WorkflowExecutionError) as exc_info:
        run_signal(
            drug_id="semaglutide",
            event_id="pancreatitis",
            signal_metrics=_minimal_signal_metrics(),
            evidence_window="2024Q1",
            paths=isolated_workflow,
        )

    assert "Trial data load error" in str(exc_info.value)
    assert exc_info.value.errors
    report_path = isolated_workflow.processed_reports / "semaglutide__pancreatitis_report.md"
    assert not report_path.exists()
