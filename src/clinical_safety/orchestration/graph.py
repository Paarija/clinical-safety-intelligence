"""
orchestration/graph.py

Full LangGraph evidence workflow for the Clinical Safety Intelligence System.

Implements the 10-node graph specified in docs/langgraph_workflow.md:

  1.  candidate_signal_selector     — initialize one ranked signal
  2.  faers_evidence_builder        — build FAERS summary (no LLM)
  3.  trial_evidence_builder        — build trial summary (no LLM)
  4.  regulatory_evidence_retriever — FDA label + safety comms
  5.  literature_evidence_retriever — PubMed
  6.  evidence_quality_gate         — deterministic metadata checks
  7.  evidence_synthesizer          — Gemini synthesis (guardrailed)
  8.  evidence_grader               — deterministic A/B/C/D grading
  9.  human_review                  — record analyst-review requirement
  10. report_generator              — renders EvidencePacket to Markdown

Conditional edges:
  - After quality gate: if no accepted evidence → skip synthesizer, goto grader
  - After grader: if human_review_required → add review note, then report
  - Otherwise: report directly

Usage:
    Use run_evidence_workflow.py for CLI execution over ranked candidate signals.

    from clinical_safety.orchestration.graph import build_graph, run_signal
    # run_signal(...) expects SignalMetrics from signal_ranking.py.
"""

from __future__ import annotations
import inspect

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from clinical_safety.common.exceptions import WorkflowExecutionError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import (
    ArmType,
    EvidenceDocument,
    EvidencePacket,
    SignalMetrics,
    TrialAdverseEvent,
)
from clinical_safety.evidence.retrievers.fda_retriever import FDARetriever
from clinical_safety.evidence.retrievers.pubmed_retriever import PubMedRetriever
from clinical_safety.llm.gemini_provider import GeminiProvider
from clinical_safety.llm.guardrails import Guardrails
from clinical_safety.llm.prompts.synthesis_prompt import (
    build_synthesis_messages,
    parse_synthesis_output,
)
from clinical_safety.orchestration.grader import grade_evidence, requires_human_review

logger = get_logger(__name__)


# ── Graph State ───────────────────────────────────────────────────────────────

class SafetyIntelligenceState(TypedDict, total=False):
    """
    Carries all signal evidence through the graph.
    Fields are populated progressively by each node.
    """
    # Signal identity
    drug_id: str
    event_id: str
    signal_id: str
    drug_label: str
    event_label: str
    evidence_window: str
    event_retrieval_terms: list[str]

    # Assembled EvidencePacket (built up across nodes)
    packet: EvidencePacket

    # Error / status messages
    errors: list[str]
    workflow_complete: bool
    human_review_required: bool

    # Optional: paths for I/O
    paths: Paths


# ── Helper: drug/event label lookup ──────────────────────────────────────────

def _get_drug_label(drug_id: str) -> str:
    try:
        from clinical_safety.common.config import get_config
        cfg = get_config()
        for drug in cfg.drug_scope.drugs:
            if drug.id == drug_id:
                return drug.normalized_name.title()
    except (KeyError, AttributeError, ImportError):
        logger.warning("Could not resolve label for drug_id=%r from config, using fallback", drug_id)
    return drug_id.replace("_", " ").title()


def _get_event_label(event_id: str) -> str:
    try:
        from clinical_safety.common.config import get_config
        cfg = get_config()
        for fam in cfg.event_scope.event_families:
            if fam.id == event_id:
                return fam.label
    except (KeyError, AttributeError, ImportError):
        logger.warning("Could not resolve label for event_id=%r from config, using fallback", event_id)
    return event_id.replace("_", " ").title()

def _normalize_retrieval_terms(terms: list[str] | None) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        if not isinstance(raw, str):
            continue
        term = raw.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def _get_event_retrieval_terms(event_id: str, fallback_event_label: str) -> list[str]:
    try:
        from clinical_safety.common.config import get_config
        cfg = get_config()
        for fam in cfg.event_scope.event_families:
            if fam.id == event_id:
                preferred_terms = _normalize_retrieval_terms(fam.preferred_terms)
                if preferred_terms:
                    return preferred_terms
    except (KeyError, AttributeError, ImportError):
        logger.warning(
            "Could not resolve retrieval terms for event_id=%r from config, using fallback",
            event_id,
        )
    return _normalize_retrieval_terms([fallback_event_label])


def _retriever_accepts_event_terms(retriever: object) -> bool:
    try:
        signature = inspect.signature(getattr(retriever, "retrieve"))
    except (TypeError, ValueError):
        return False
    return "event_terms" in signature.parameters


def _call_retriever(
    retriever: object,
    *,
    drug_id: str,
    event_id: str,
    drug_name: str,
    event_name: str,
    event_terms: list[str],
):
    if _retriever_accepts_event_terms(retriever):
        return retriever.retrieve(
            drug_id=drug_id,
            event_id=event_id,
            drug_name=drug_name,
            event_name=event_name,
            event_terms=event_terms,
        )
    return retriever.retrieve(
        drug_id=drug_id,
        event_id=event_id,
        drug_name=drug_name,
        event_name=event_name,
    )


# ── Node 1: Candidate signal selector ────────────────────────────────────────

def candidate_signal_selector(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 1: Initialise the EvidencePacket from the signal row already in state.

    In full pipeline use, a caller injects drug_id + event_id + metrics into
    state before invoking the graph. This node creates the EvidencePacket shell.
    """
    drug_id = state.get("drug_id", "unknown")
    event_id = state.get("event_id", "unknown")
    signal_id = f"{drug_id}__{event_id}"
    drug_label = _get_drug_label(drug_id)
    event_label = _get_event_label(event_id)
    event_retrieval_terms = _get_event_retrieval_terms(event_id, event_label)
    evidence_window = state.get("evidence_window", "unknown")

    old_packet = state.get("packet")
    metrics = old_packet.signal_metrics if old_packet else None

    packet = EvidencePacket(
        signal_id=signal_id,
        drug_id=drug_id,
        event_id=event_id,
        drug_label=drug_label,
        event_label=event_label,
        evidence_window=evidence_window,
        signal_metrics=metrics,
    )

    logger.info("Graph node 1 — signal selected: %s", signal_id)
    return {
        **state,
        "packet": packet,
        "signal_id": signal_id,
        "drug_label": drug_label,
        "event_label": event_label,
        "event_retrieval_terms": event_retrieval_terms,
        "errors": [],
    }


# ── Node 2: FAERS evidence builder ───────────────────────────────────────────

def faers_evidence_builder(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 2: Attach FAERS signal metrics to the EvidencePacket.

    The signal_metrics are expected to be pre-attached (from the ranked signal
    shortlist). This node validates presence and logs a summary.
    No LLM call is made here.
    """
    packet: EvidencePacket = state["packet"]

    if packet.signal_metrics is None:
        logger.warning("Node 2: no signal_metrics in packet for %s", packet.signal_id)
        errors = list(state.get("errors", []))
        errors.append("FAERS signal metrics not available")
        return {**state, "errors": errors}

    m = packet.signal_metrics
    logger.info(
        "Graph node 2 — FAERS: signal=%s, cases=%d, ROR=%.2f (CI %.2f–%.2f), "
        "deaths=%d, hosp=%d",
        packet.signal_id, m.case_count,
        m.ror or 0, m.ror_lower_ci or 0, m.ror_upper_ci or 0,
        m.death_count, m.hospitalization_count,
    )
    return state


# ── Node 3: Trial evidence builder ───────────────────────────────────────────

def trial_evidence_builder(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 3: Load trial AE data from trial_comparison.parquet for this signal.

    If trial data is missing → mark trial_evidence_available=False and add
    evidence gap. Never raises — missing trial data is an explicit data state.
    """
    packet: EvidencePacket = state["packet"]
    paths: Paths = state.get("paths") or Paths()

    trial_file = paths.processed_analytics / "trial_comparison.parquet"
    if not trial_file.exists():
        logger.info(
            "Node 3: no trial_comparison.parquet found — marking trial evidence unavailable"
        )
        packet.evidence_gaps.append(
            "No ClinicalTrials.gov comparison data available "
            "(run ClinicalTrialsSource + TrialComparator first)"
        )
        return state

    try:
        trial_df = pd.read_parquet(trial_file)
        # Filter to this signal's drug+event
        signal_df = trial_df[
            (trial_df["drug_id"] == packet.drug_id) &
            (trial_df["event_id"] == packet.event_id)
        ]

        if signal_df.empty:
            logger.info(
                "Node 3: no trial rows for %s+%s", packet.drug_id, packet.event_id
            )
            packet.evidence_gaps.append(
                f"No trial adverse-event data found for "
                f"{packet.drug_label} + {packet.event_label}"
            )
        else:
            packet.trial_evidence_available = True
            # Convert to TrialAdverseEvent objects
            for _, row in signal_df.iterrows():
                tae = TrialAdverseEvent(
                    nct_id=str(row.get("nct_id", "unknown")),
                    arm_id=str(row.get("arm_label", "unknown")),
                    arm_type=ArmType(str(row.get("arm_type", "unknown"))),
                    event_term=str(row.get("event_term", packet.event_label)),
                    event_id=packet.event_id,
                    affected_participants=_safe_int_or_none(row.get("affected")),
                    at_risk_participants=_safe_int_or_none(row.get("at_risk")),
                    event_rate=_safe_float_or_none(row.get("event_rate")),
                    is_serious=bool(row.get("is_serious", False)),
                )
                packet.trial_ae_rates.append(tae)

            # Check for contradiction: trial rate much lower than expected from FAERS signal
            _check_trial_contradiction(packet)
            logger.info(
                "Node 3: %d trial AE rows loaded for %s", len(packet.trial_ae_rates), packet.signal_id
            )
    except Exception as exc:
        logger.warning("Node 3: trial data load error: %s", exc)
        message = f"Trial data load error: {exc}"
        packet.evidence_gaps.append(message)
        errors = list(state.get("errors", []))
        errors.append(message)
        return {**state, "errors": errors}

    return state


def _check_trial_contradiction(packet: EvidencePacket) -> None:
    """
    Mark a synthesis contradiction if trial data shows zero/very low event rate
    while FAERS shows a strong signal. This is a simple heuristic.
    """
    treatment_rates = [
        tae.event_rate for tae in packet.trial_ae_rates
        if tae.arm_type is ArmType.TREATMENT and tae.event_rate is not None
    ]
    if not treatment_rates:
        return

    avg_trial_rate = sum(treatment_rates) / len(treatment_rates)
    if avg_trial_rate == 0 and packet.signal_metrics and packet.signal_metrics.case_count >= 10:
        packet.synthesis_contradictions.append(
            f"Trial data shows zero event rate for {packet.event_label} "
            f"in treatment arm, while FAERS shows {packet.signal_metrics.case_count} reports "
            f"(ROR lower CI = {packet.signal_metrics.ror_lower_ci}). "
            "Possible reporting bias or trial exclusion criteria differences."
        )


# ── Node 4: Regulatory evidence retriever ────────────────────────────────────

def regulatory_evidence_retriever(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 4: Retrieve FDA label and safety communication evidence.

    Uses FDARetriever. Failures are caught and logged as evidence gaps.
    """
    packet: EvidencePacket = state["packet"]
    event_terms = state.get("event_retrieval_terms", [packet.event_label])
    try:
        retriever = FDARetriever()
        docs = _call_retriever(
            retriever,
            drug_id=packet.drug_id,
            event_id=packet.event_id,
            drug_name=packet.drug_label,
            event_name=packet.event_label,
            event_terms=event_terms,
        )
        label_docs = [d for d in docs if d.source_type == "fda_label"]
        safety_docs = [d for d in docs if d.source_type != "fda_label"]
        packet.label_documents.extend(label_docs)
        packet.regulatory_documents.extend(safety_docs)
        logger.info(
            "Node 4: FDA retrieval complete — %d label, %d safety comm docs",
            len(label_docs), len(safety_docs),
        )
    except Exception as exc:
        logger.warning("Node 4: FDA retrieval error: %s", exc)
        message = f"FDA evidence retrieval failed: {exc}"
        packet.evidence_gaps.append(message)
        errors = list(state.get("errors", []))
        errors.append(message)
        return {**state, "errors": errors}

    return state


# ── Node 5: Literature evidence retriever ────────────────────────────────────

def literature_evidence_retriever(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 5: Retrieve PubMed literature evidence.

    Uses PubMedRetriever. Failures are caught and logged as evidence gaps.
    """
    packet: EvidencePacket = state["packet"]
    event_terms = state.get("event_retrieval_terms", [packet.event_label])
    try:
        retriever = PubMedRetriever()
        docs = _call_retriever(
            retriever,
            drug_id=packet.drug_id,
            event_id=packet.event_id,
            drug_name=packet.drug_label,
            event_name=packet.event_label,
            event_terms=event_terms,
        )
        packet.literature_documents.extend(docs)
        logger.info(
            "Node 5: PubMed retrieval complete — %d documents", len(docs)
        )
    except Exception as exc:
        logger.warning("Node 5: PubMed retrieval error: %s", exc)
        message = f"PubMed evidence retrieval failed: {exc}"
        packet.evidence_gaps.append(message)
        errors = list(state.get("errors", []))
        errors.append(message)
        return {**state, "errors": errors}

    return state


# ── Node 6: Evidence quality gate ────────────────────────────────────────────

def evidence_quality_gate(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 6: Deterministic quality gate over all retrieved documents.

    Checks each document for:
      - Non-empty title and snippet
      - Relevance: drug name or event name appears in title or snippet
      - Minimum snippet length (> 20 chars)

    Rejected documents are moved to packet.rejected_evidence.
    Documents accepted here are the only ones the synthesizer sees.
    """
    packet: EvidencePacket = state["packet"]
    drug_lower = packet.drug_label.lower()
    event_terms = _normalize_retrieval_terms(state.get("event_retrieval_terms", [packet.event_label]))
    event_terms_lower = [term.lower() for term in event_terms]

    def check_doc(doc: EvidenceDocument) -> bool:
        """Return True if the document passes quality checks."""
        content = ((doc.title or "") + " " + (doc.snippet or "")).lower()
        if not content.strip():
            doc.accepted = False
            doc.rejection_reason = "empty content"
            return False
        if len(doc.snippet or "") < 20 and not doc.title:
            doc.accepted = False
            doc.rejection_reason = "snippet too short and no title"
            return False
        has_event_term = any(term in content for term in event_terms_lower)
        if doc.source_type == "fda_safety_comm" and not has_event_term:
            doc.accepted = False
            doc.rejection_reason = "FDA safety communication missing event term"
            return False
        # Relevance: either drug or any retrieval term appears in content
        if drug_lower not in content and not has_event_term:
            doc.accepted = False
            doc.rejection_reason = "not relevant to drug or event"
            return False
        return True

    rejected: list[EvidenceDocument] = []
    for doc_list in [
        packet.regulatory_documents,
        packet.literature_documents,
        packet.label_documents,
    ]:
        for doc in doc_list:
            if not check_doc(doc):
                rejected.append(doc)

    packet.rejected_evidence.extend(rejected)

    total = (
        len(packet.regulatory_documents)
        + len(packet.literature_documents)
        + len(packet.label_documents)
    )
    accepted = len(packet.all_accepted_documents)
    logger.info(
        "Node 6: quality gate — %d/%d documents accepted, %d rejected",
        accepted, total, len(rejected),
    )

    if total == 0:
        packet.evidence_gaps.append(
            "No external evidence documents were retrieved for this signal."
        )
    elif accepted == 0:
        packet.evidence_gaps.append(
            "All retrieved documents were rejected by the quality gate. "
            "No external evidence available for synthesis."
        )

    return state


# ── Routing function after quality gate ──────────────────────────────────────

def _route_after_quality_gate(state: SafetyIntelligenceState) -> str:
    """
    Conditional edge: if no accepted evidence, skip synthesizer and go straight to grader.
    """
    packet: EvidencePacket = state["packet"]
    if not packet.all_accepted_documents:
        logger.info("Routing: no accepted evidence — skipping synthesizer")
        return "evidence_grader"
    return "evidence_synthesizer"


# ── Node 7: Evidence synthesizer ─────────────────────────────────────────────

def evidence_synthesizer(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 7: Call Gemini to synthesize evidence. Output is guardrail-checked.

    In DRY_RUN mode, returns the GeminiProvider placeholder.
    The parsed output populates packet.synthesis_summary,
    packet.synthesis_supports, and packet.synthesis_contradictions.
    """
    packet: EvidencePacket = state["packet"]
    try:
        provider = GeminiProvider()
        guardrail = Guardrails(provider)
        messages = build_synthesis_messages(packet)
        raw_text = provider.invoke(messages)
        safe_text = guardrail.check_and_correct(raw_text, messages)
        parsed = parse_synthesis_output(safe_text)

        packet.synthesis_summary = str(parsed.get("synthesis_summary", ""))
        supports = parsed.get("supporting_evidence", [])
        if isinstance(supports, list):
            packet.synthesis_supports = [str(s) for s in supports]
        contras = parsed.get("contradictions", [])
        if isinstance(contras, list):
            # Merge with any existing contradictions (e.g. from trial comparison)
            packet.synthesis_contradictions = (
                packet.synthesis_contradictions + [str(c) for c in contras
                                                   if c != "None identified"]
            )
        packet.limitation_statement = str(parsed.get("limitation_statement", ""))

        logger.info(
            "Node 7: synthesis complete — summary length=%d, supports=%d, contradictions=%d",
            len(packet.synthesis_summary),
            len(packet.synthesis_supports),
            len(packet.synthesis_contradictions),
        )
    except Exception as exc:
        logger.error("Node 7: synthesis failed: %s", exc)
        packet.synthesis_summary = (
            "[Synthesis failed — manual analyst review required.]"
        )
        errors = list(state.get("errors", []))
        errors.append(f"Synthesis error: {exc}")
        return {**state, "errors": errors}

    return state


# ── Node 8: Evidence grader ───────────────────────────────────────────────────

def evidence_grader(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 8: Apply deterministic A/B/C/D grading rubric.

    Populates packet.evidence_grade, packet.triage_status, and
    packet.human_review_required.
    """
    packet: EvidencePacket = state["packet"]
    grade, explanation = grade_evidence(packet)
    packet.grade_explanation = explanation
    packet.human_review_required = requires_human_review(packet)

    logger.info(
        "Node 8: grade=%s, triage=%s, human_review=%s",
        grade.value, packet.triage_status.value, packet.human_review_required,
    )
    return {**state, "human_review_required": packet.human_review_required}


# ── Routing after grader ──────────────────────────────────────────────────────

def _route_after_grader(state: SafetyIntelligenceState) -> str:
    """
    Conditional edge: Grade A or serious contradictions → human_review checkpoint.
    Otherwise go directly to report_generator.
    """
    if state.get("human_review_required"):
        logger.info("Routing: human review required — routing to human_review node")
        return "human_review"
    return "report_generator"


# ── Node 8b: Human review checkpoint ─────────────────────────────────────────

def human_review(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 8b: Human review checkpoint (pause point).

    In automated runs, this node logs the requirement and adds a note to the
    packet. In a Streamlit workflow, an analyst would review and approve here.
    For now, the graph continues automatically but marks the packet clearly.
    """
    packet: EvidencePacket = state["packet"]
    packet.human_review_notes = (
        f"Human review required for signal {packet.signal_id} "
        f"(grade={packet.evidence_grade.value if packet.evidence_grade else 'pending'}). "
        f"Reason: {packet.grade_explanation or 'see triage status'}. "
        f"This report was generated automatically and requires analyst sign-off before use."
    )
    logger.info(
        "Node 8b: human review checkpoint reached for %s — proceeding to report generation",
        packet.signal_id,
    )
    return state


# ── Node 9: Report generator ──────────────────────────────────────────────────

def report_generator(state: SafetyIntelligenceState) -> SafetyIntelligenceState:
    """
    Node 9: Generate the final Markdown report from the completed EvidencePacket.

    Delegates to reporting/report_generator.py. Saves the report to
    data/processed/reports/<signal_id>_report.md and marks packet.report_generated = True.
    """
    packet: EvidencePacket = state["packet"]
    if state.get("errors"):
        logger.warning("Node 9: skipping report generation because workflow has errors")
        return state
    try:
        from clinical_safety.reporting.report_generator import ReportGenerator
        from clinical_safety.common.mock_data import is_mock_pipeline_data
        paths: Paths = state.get("paths") or Paths()
        if is_mock_pipeline_data(paths):
            logger.warning("Node 9: skipping report generation because pipeline data is marked synthetic")
            return state
        rg = ReportGenerator(paths=paths)
        report_path = rg.generate(packet)
        packet.report_generated = True
        logger.info("Node 9: report saved -> %s", report_path)
    except Exception as exc:
        logger.error("Node 9: report generation failed: %s", exc)
        errors = list(state.get("errors", []))
        errors.append(f"Report generation error: {exc}")
        return {**state, "errors": errors}

    return {**state, "workflow_complete": True}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Returns:
        A compiled LangGraph runnable (CompiledGraph).

    Raises:
        ImportError: If langgraph is not installed.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph is not installed. Run: pip install langgraph"
        ) from exc

    workflow = StateGraph(SafetyIntelligenceState)

    # Register nodes
    workflow.add_node("candidate_signal_selector", candidate_signal_selector)
    workflow.add_node("faers_evidence_builder", faers_evidence_builder)
    workflow.add_node("trial_evidence_builder", trial_evidence_builder)
    workflow.add_node("regulatory_evidence_retriever", regulatory_evidence_retriever)
    workflow.add_node("literature_evidence_retriever", literature_evidence_retriever)
    workflow.add_node("evidence_quality_gate", evidence_quality_gate)
    workflow.add_node("evidence_synthesizer", evidence_synthesizer)
    workflow.add_node("evidence_grader", evidence_grader)
    workflow.add_node("human_review", human_review)
    workflow.add_node("report_generator", report_generator)

    # Linear edges (no branching)
    workflow.set_entry_point("candidate_signal_selector")
    workflow.add_edge("candidate_signal_selector", "faers_evidence_builder")
    workflow.add_edge("faers_evidence_builder", "trial_evidence_builder")
    workflow.add_edge("trial_evidence_builder", "regulatory_evidence_retriever")
    workflow.add_edge("regulatory_evidence_retriever", "literature_evidence_retriever")
    workflow.add_edge("literature_evidence_retriever", "evidence_quality_gate")

    # Conditional edge after quality gate
    workflow.add_conditional_edges(
        "evidence_quality_gate",
        _route_after_quality_gate,
        {
            "evidence_synthesizer": "evidence_synthesizer",
            "evidence_grader": "evidence_grader",
        },
    )
    workflow.add_edge("evidence_synthesizer", "evidence_grader")

    # Conditional edge after grader
    workflow.add_conditional_edges(
        "evidence_grader",
        _route_after_grader,
        {
            "human_review": "human_review",
            "report_generator": "report_generator",
        },
    )
    workflow.add_edge("human_review", "report_generator")
    workflow.add_edge("report_generator", END)
    # Configure persistent checkpointer for human_review node
    import os as _os
    checkpointer = None
    checkpoint_dir = _os.getenv("CHECKPOINT_DIR")
    if checkpoint_dir:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            configured_dir = Path(checkpoint_dir)
            db_dir = (
                configured_dir
                if configured_dir.is_absolute()
                else Path(__file__).resolve().parents[3] / configured_dir
            )
            db_dir.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(str(db_dir / "checkpoints.db"))
            logger.info("LangGraph: using SqliteSaver at %s", db_dir / "checkpoints.db")
        except ImportError:
            logger.warning(
                "CHECKPOINT_DIR set but langgraph-checkpoint-sqlite not installed. "
                "Run: pip install langgraph-checkpoint-sqlite"
            )

    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph: workflow compiled successfully (10 nodes)")

    if not checkpointer:
        logger.warning(
            "No checkpointer configured — human_review state will be lost on restart. "
            "Set CHECKPOINT_DIR env var and install langgraph-checkpoint-sqlite."
        )

    return compiled


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_signal(
    drug_id: str,
    event_id: str,
    signal_metrics: SignalMetrics | None = None,
    evidence_window: str = "unknown",
    paths: Paths | None = None,
    graph=None,
) -> EvidencePacket:
    """
    Run the full workflow for a single drug-event signal.

    Args:
        drug_id         : Internal drug identifier.
        event_id        : Internal event identifier.
        signal_metrics  : Pre-computed SignalMetrics (from signal_ranking.py).
        evidence_window : Label for the FAERS quarter.
        paths           : Paths resolver instance.
        graph           : Pre-compiled graph (built once and reused).

    Returns:
        Completed EvidencePacket with grade, synthesis, and report.
    """
    if graph is None:
        graph = build_graph()

    initial_state: SafetyIntelligenceState = {
        "drug_id": drug_id,
        "event_id": event_id,
        "evidence_window": evidence_window,
        "paths": paths or Paths(),
        "errors": [],
        "workflow_complete": False,
        "human_review_required": False,
    }

    # Pre-attach signal metrics to the packet (will be pulled in node 2)
    # We seed the packet here so node 1 can find it
    dummy_packet = EvidencePacket(
        signal_id=f"{drug_id}__{event_id}",
        drug_id=drug_id,
        event_id=event_id,
        drug_label=_get_drug_label(drug_id),
        event_label=_get_event_label(event_id),
        evidence_window=evidence_window,
        signal_metrics=signal_metrics,
    )
    initial_state["packet"] = dummy_packet  # type: ignore[assignment]

    final_state = graph.invoke(initial_state)
    errors = final_state.get("errors", [])
    complete = final_state.get("workflow_complete", False)
    packet = final_state.get("packet", dummy_packet)
    if errors or not complete:
        raise WorkflowExecutionError(packet.signal_id, list(errors))
    return packet


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int_or_none(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
