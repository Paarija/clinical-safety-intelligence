"""
llm/prompts/synthesis_prompt.py

Prompt templates for the evidence synthesizer node in the
Clinical Safety Intelligence LangGraph workflow.

The synthesizer receives an EvidencePacket and must produce:
  1. A synthesis summary paragraph (3-5 sentences, hedged language, cited)
  2. A list of supporting evidence points
  3. A list of contradictions / gaps

CRITICAL CONSTRAINTS (enforced by Guardrails, also stated in prompt):
  - No causal language ("causes", "proves", "definitively")
  - No clinical recommendations (prescribing, withdrawal, treatment)
  - No diagnosis claims
  - Every factual claim must reference a source

Usage:
    from clinical_safety.llm.prompts.synthesis_prompt import build_synthesis_messages
    messages = build_synthesis_messages(packet)
    raw_text = provider.invoke(messages)
    safe_text = guardrails.check_and_correct(raw_text, messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clinical_safety.common.types import EvidencePacket

_SYSTEM_PROMPT = """You are a pharmacovigilance signal triage analyst working for a \
regulatory affairs team. Your role is to synthesize evidence about a potential \
drug-adverse event signal and produce a structured summary for analyst review.

STRICT RULES — Violation will cause your response to be rejected:
1. Do NOT claim the drug causes the adverse event. Use hedged language: \
"reports suggest an association", "is associated with", "an elevated reporting rate was observed".
2. Do NOT make clinical recommendations (prescribing, withdrawal, dose adjustment, diagnosis, treatment).
3. Do NOT use "proves", "definitively causes", "drug should be withdrawn", \
"recommend discontinuing", "treat the patient", "clinical recommendation".
4. Every factual claim you make MUST reference one of the provided sources \
using [SOURCE_ID] citation format.
5. Acknowledge limitations and data gaps explicitly.
6. Your output is for analyst review only — it is NOT a medical or regulatory decision.

Output format (return this structure only):

SYNTHESIS_SUMMARY:
<3-5 sentence hedged synthesis paragraph with inline [SOURCE_ID] citations>

SUPPORTING_EVIDENCE:
- <point 1 [SOURCE_ID]>
- <point 2 [SOURCE_ID]>

CONTRADICTIONS_AND_GAPS:
- <item 1>
- <item 2 (or "None identified" if none)>

LIMITATION_STATEMENT:
<1-2 sentence statement of key analytical limitations>
"""


def build_synthesis_messages(packet: "EvidencePacket") -> list[dict[str, str]]:
    """
    Build the LLM message list for the evidence synthesizer node.

    Args:
        packet: The EvidencePacket assembled by earlier graph nodes.

    Returns:
        List of role/content dicts for GeminiProvider.invoke().
    """
    human_content = _build_human_block(packet)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "human", "content": human_content},
    ]


def _build_human_block(packet: "EvidencePacket") -> str:
    """
    Assemble the human-turn prompt block from the EvidencePacket contents.
    All factual data is injected as structured text — the LLM does NOT
    retrieve data itself.
    """
    parts: list[str] = []

    # ── Signal header ─────────────────────────────────────────────────────────
    parts.append(
        f"SIGNAL: {packet.drug_label} + {packet.event_label}\n"
        f"Signal ID: {packet.signal_id}\n"
        f"Evidence Window: {packet.evidence_window}\n"
    )

    # ── FAERS quantitative evidence ───────────────────────────────────────────
    if packet.signal_metrics:
        m = packet.signal_metrics
        parts.append(
            "FAERS DISPROPORTIONALITY EVIDENCE:\n"
            f"  Case count: {m.case_count}\n"
            f"  ROR: {m.ror} (95% CI: {m.ror_lower_ci} – {m.ror_upper_ci})\n"
            f"  PRR: {m.prr}\n"
            f"  Chi2 p-value: {m.chi2_p_value}\n"
            f"  Seriousness rate: {f'{m.seriousness_rate:.1%}' if m.seriousness_rate is not None else 'N/A'} of cases with serious outcome\n"
            f"  Deaths: {m.death_count}, Hospitalisations: {m.hospitalization_count}\n"
            "  [FAERS reports cannot establish causality. These are reporting rates.]\n"
        )
    else:
        parts.append("FAERS DISPROPORTIONALITY EVIDENCE: Not available.\n")

    # ── Trial evidence ────────────────────────────────────────────────────────
    if packet.trial_evidence_available and packet.trial_ae_rates:
        parts.append("CLINICAL TRIAL EVIDENCE:")
        for ae in packet.trial_ae_rates[:10]:   # cap at 10 rows to avoid token overflow
            if ae.event_id == packet.event_id or ae.event_id is None:
                parts.append(
                    f"  [TRIAL:{ae.nct_id}] arm={ae.arm_id} "
                    f"event='{ae.event_term}' "
                    f"rate={f'{ae.event_rate:.4f}' if ae.event_rate is not None else 'N/A'} "
                    f"serious={'yes' if ae.is_serious else 'no'}"
                )
        parts.append("")
    else:
        parts.append("CLINICAL TRIAL EVIDENCE: Not available for this signal.\n")

    # ── Regulatory / label documents ──────────────────────────────────────────
    reg_docs = [d for d in packet.regulatory_documents + packet.label_documents if d.accepted]
    if reg_docs:
        parts.append("REGULATORY AND LABEL EVIDENCE:")
        for doc in reg_docs:
            src_id = doc.doc_id
            parts.append(
                f"  [{src_id}] {doc.source_type.upper()}: {doc.title}\n"
                f"    URL: {doc.url or 'N/A'}\n"
                f"    Snippet: {doc.snippet[:400] if doc.snippet else 'No snippet.'}"
            )
        parts.append("")
    else:
        parts.append("REGULATORY AND LABEL EVIDENCE: None retrieved.\n")

    # ── Literature documents ──────────────────────────────────────────────────
    lit_docs = [d for d in packet.literature_documents if d.accepted]
    if lit_docs:
        parts.append("LITERATURE EVIDENCE:")
        for doc in lit_docs:
            src_id = doc.doc_id
            parts.append(
                f"  [{src_id}] PUBMED ({doc.identifier or 'PMID unknown'}): {doc.title}\n"
                f"    Snippet: {doc.snippet[:400] if doc.snippet else 'No abstract.'}"
            )
        parts.append("")
    else:
        parts.append("LITERATURE EVIDENCE: None retrieved.\n")

    # ── Evidence gaps ─────────────────────────────────────────────────────────
    if packet.evidence_gaps:
        parts.append("EVIDENCE GAPS (identified by quality gate):")
        for gap in packet.evidence_gaps:
            parts.append(f"  - {gap}")
        parts.append("")

    parts.append(
        "Please synthesize all of the above evidence following the output format and rules "
        "specified in your system instructions."
    )

    return "\n".join(parts)


def parse_synthesis_output(raw_text: str) -> dict[str, str | list[str]]:
    """
    Parse the structured LLM output back into a dict.

    Returns:
        {
          'synthesis_summary': str,
          'supporting_evidence': list[str],
          'contradictions': list[str],
          'limitation_statement': str,
        }
    All fields default to empty if the section is missing.
    """
    result: dict[str, str | list[str]] = {
        "synthesis_summary": "",
        "supporting_evidence": [],
        "contradictions": [],
        "limitation_statement": "",
    }

    section = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("SYNTHESIS_SUMMARY:"):
            section = "summary"
        elif stripped.startswith("SUPPORTING_EVIDENCE:"):
            section = "support"
        elif stripped.startswith("CONTRADICTIONS_AND_GAPS:"):
            section = "contra"
        elif stripped.startswith("LIMITATION_STATEMENT:"):
            section = "limit"
        elif section == "summary" and stripped:
            result["synthesis_summary"] = (
                str(result["synthesis_summary"]) + (" " if result["synthesis_summary"] else "") + stripped
            )
        elif section == "support" and stripped.startswith("- "):
            cast = result["supporting_evidence"]
            if isinstance(cast, list):
                cast.append(stripped[2:])
        elif section == "contra" and stripped.startswith("- "):
            cast = result["contradictions"]
            if isinstance(cast, list):
                cast.append(stripped[2:])
        elif section == "limit" and stripped:
            result["limitation_statement"] = (
                str(result["limitation_statement"]) + (" " if result["limitation_statement"] else "") + stripped
            )

    return result
