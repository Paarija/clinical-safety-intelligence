"""
reporting/report_generator.py

Renders a completed EvidencePacket into a traceable Markdown report.

Design principles:
  - Retrieved evidence is listed with source identifiers
  - Limitation banner is always present
  - Evidence grade is prominently displayed
  - Missing evidence is stated explicitly — never silently omitted
  - Reports are saved to data/processed/reports/<signal_id>_report.md

Usage:
    from clinical_safety.reporting.report_generator import ReportGenerator
    rg = ReportGenerator()
    report_path = rg.generate(packet)
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from pathlib import Path

from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import EvidenceGrade, EvidencePacket

logger = get_logger(__name__)

_GRADE_LABELS = {
    EvidenceGrade.A: "Grade A - Strong Signal for Review",
    EvidenceGrade.B: "Grade B - Moderate Signal",
    EvidenceGrade.C: "Grade C - Weak or Uncertain Signal",
    EvidenceGrade.D: "Grade D - Insufficient Evidence",
}

_LIMITATION_BANNER = textwrap.dedent("""\
    > **IMPORTANT LIMITATIONS**
    > This report is a safety-signal triage output for analyst review.
    > FAERS spontaneous reports cannot establish causality. All disproportionality metrics
    > (ROR, PRR) are reporting-rate measures — not incidence or risk estimates.
    > Outputs require independent expert review before use in clinical, prescribing,
    > regulatory, or operational safety decisions.
""")


class ReportGenerator:
    """
    Generates Markdown safety signal reports from EvidencePacket objects.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        self._paths = paths or Paths()
        self._reports_dir = self._paths.processed_reports
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, packet: EvidencePacket, save: bool = True) -> Path:
        """
        Render EvidencePacket to Markdown and optionally save.

        Args:
            packet: A fully graded EvidencePacket.
            save  : Write report to disk.

        Returns:
            Path to the saved report file.
        """
        md = self._render(packet)
        out_path = self._reports_dir / f"{packet.signal_id}_report.md"

        if save:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            logger.info("Report saved -> %s", out_path)

        return out_path

    def render_string(self, packet: EvidencePacket) -> str:
        """Return the Markdown report as a string without saving."""
        return self._render(packet)

    def _render(self, packet: EvidencePacket) -> str:
        """Assemble the full Markdown document."""
        sections: list[str] = []

        sections.append(self._render_header(packet))
        sections.append(_LIMITATION_BANNER)
        sections.append(self._render_grade_badge(packet))
        sections.append(self._render_signal_overview(packet))
        sections.append(self._render_faers_section(packet))
        sections.append(self._render_trial_section(packet))
        sections.append(self._render_synthesis_section(packet))
        sections.append(self._render_evidence_gaps(packet))

        if packet.human_review_notes:
            sections.append(self._render_human_review_note(packet))

        sections.append(self._render_bibliography(packet))
        sections.append(self._render_footer(packet))

        return "\n\n---\n\n".join(s for s in sections if s.strip())

    @staticmethod
    def _render_header(packet: EvidencePacket) -> str:
        return (
            f"# Safety Signal Report: {packet.drug_label} + {packet.event_label}\n\n"
            f"**Signal ID:** `{packet.signal_id}`  \n"
            f"**Evidence Window:** {packet.evidence_window}  \n"
            f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}  \n"
            f"**System:** Clinical Safety Intelligence v0.1 (Analyst Review Workflow)"
        )

    @staticmethod
    def _render_grade_badge(packet: EvidencePacket) -> str:
        grade = packet.evidence_grade
        label = _GRADE_LABELS.get(grade, "Grade Unknown - Pending")
        triage = packet.triage_status.value.replace("_", " ").title() if packet.triage_status else "Pending"
        explanation = packet.grade_explanation or "No grading explanation available."
        return (
            f"## Evidence Grade\n\n"
            f"### {label}\n\n"
            f"**Triage Status:** {triage}  \n"
            f"**Grade Rationale:** {explanation}"
        )

    @staticmethod
    def _render_signal_overview(packet: EvidencePacket) -> str:
        m = packet.signal_metrics
        if not m:
            return "## Signal Overview\n\n*FAERS signal metrics not available.*"

        return (
            f"## Signal Overview\n\n"
            f"| Metric | Value |\n"
            f"|:---|:---|\n"
            f"| Drug | {packet.drug_label} |\n"
            f"| Adverse Event | {packet.event_label} |\n"
            f"| FAERS Case Count | {m.case_count} |\n"
            f"| ROR | {m.ror:.3f} |\n"
            f"| ROR 95% CI | {m.ror_lower_ci:.3f} – {m.ror_upper_ci:.3f} |\n"
            f"| PRR | {f'{m.prr:.3f}' if m.prr is not None else 'N/A'} |\n"
            f"| Chi² p-value | {f'{m.chi2_p_value:.4f}' if m.chi2_p_value is not None else 'N/A'} |\n"
            f"| Seriousness Rate | {m.seriousness_rate:.1%} |\n"
            f"| Deaths Reported | {m.death_count} |\n"
            f"| Hospitalisations | {m.hospitalization_count} |\n"
            f"| Drug Mapping Confidence | {m.drug_mapping_confidence.value} |\n"
            f"| Event Mapping Confidence | {m.event_mapping_confidence.value} |\n\n"
            f"> **Note:** ROR and PRR are *reporting rate ratios*, not incidence rates. "
            f"They measure whether this drug-event combination is reported "
            f"disproportionately compared to all other reports in the FAERS database. "
            f"They cannot be used to estimate the probability that a patient will "
            f"experience this adverse event."
        )

    @staticmethod
    def _render_faers_section(packet: EvidencePacket) -> str:
        m = packet.signal_metrics
        if not m:
            return "## FAERS Evidence\n\n*Not available.*"

        trend = (
            f"Trend slope: {m.trend_slope:.4f} per quarter"
            if m.trend_slope is not None
            else "Trend data not available"
        )
        publicity = (
            "Potential publicity spike detected in this period."
            if m.potential_publicity_spike
            else ""
        )

        return (
            f"## FAERS Adverse Event Reports\n\n"
            f"- {m.case_count} de-duplicated case reports identified (role: Primary Suspect)\n"
            f"- {m.death_count} cases reported with fatal outcome\n"
            f"- {m.hospitalization_count} cases with hospitalisation reported\n"
            f"- Seriousness rate: {m.seriousness_rate:.1%}\n"
            f"- {trend}\n"
            + (f"- {publicity}\n" if publicity else "")
            + f"\n*Source: FDA FAERS/AEMS Quarterly Data ({m.evidence_window})*"
        )

    @staticmethod
    def _render_trial_section(packet: EvidencePacket) -> str:
        if not packet.trial_evidence_available:
            return (
                "## Clinical Trial Evidence\n\n"
                "*No clinical trial adverse-event data was available for this signal.*  \n"
                "This does not indicate absence of a trial signal — results may not have "
                "been posted, or the event may not have been systematically collected."
            )

        # Show up to 10 AE rows
        rows = packet.trial_ae_rates[:10]
        if not rows:
            return "## Clinical Trial Evidence\n\n*Trial data available but no AE rows extracted.*"

        table = (
            "| NCT ID | Arm | Event | Rate | Serious |\n"
            "|:---|:---|:---|:---|:---|\n"
        )
        for tae in rows:
            rate_str = f"{tae.event_rate:.4f}" if tae.event_rate is not None else "N/A"
            table += (
                f"| {tae.nct_id} | {tae.arm_id} | {tae.event_term} "
                f"| {rate_str} | {'Yes' if tae.is_serious else 'No'} |\n"
            )

        # Any trial-based contradictions
        contra_note = ""
        if packet.synthesis_contradictions:
            contra_note = (
                "\n**Contradiction identified:**\n"
                + "\n".join(f"- {c}" for c in packet.synthesis_contradictions[:3])
            )

        return f"## Clinical Trial Evidence\n\n{table}{contra_note}"

    @staticmethod
    def _render_synthesis_section(packet: EvidencePacket) -> str:
        if not packet.synthesis_summary:
            return (
                "## Evidence Synthesis\n\n"
                "*Synthesis not available — either no evidence passed the quality gate "
                "or the LLM call was skipped (dry-run mode).*"
            )

        supports = (
            "\n".join(f"- {s}" for s in packet.synthesis_supports)
            if packet.synthesis_supports
            else "*None listed.*"
        )
        contras = (
            "\n".join(f"- {c}" for c in packet.synthesis_contradictions)
            if packet.synthesis_contradictions
            else "*None identified.*"
        )
        limitation = packet.limitation_statement or "*No limitation statement provided.*"

        return (
            f"## Evidence Synthesis\n\n"
            f"*Generated from the accepted evidence packet with guardrail validation. "
            f"Review cited sources and evidence gaps before using this summary.*\n\n"
            f"{packet.synthesis_summary}\n\n"
            f"### Supporting Evidence\n\n{supports}\n\n"
            f"### Contradictions and Gaps\n\n{contras}\n\n"
            f"### Analytical Limitations\n\n{limitation}"
        )

    @staticmethod
    def _render_evidence_gaps(packet: EvidencePacket) -> str:
        if not packet.evidence_gaps:
            return ""
        gaps = "\n".join(f"- {g}" for g in packet.evidence_gaps)
        return f"## Evidence Gaps\n\n{gaps}"

    @staticmethod
    def _render_human_review_note(packet: EvidencePacket) -> str:
        return (
            f"## Human Review Required\n\n"
            f"{packet.human_review_notes}"
        )

    @staticmethod
    def _render_bibliography(packet: EvidencePacket) -> str:
        all_docs = (
            packet.regulatory_documents
            + packet.literature_documents
            + packet.label_documents
        )
        if not all_docs:
            return "## References\n\n*No external evidence documents retrieved.*"

        lines = ["## References\n"]
        for doc in sorted(all_docs, key=lambda d: d.doc_id):
            status = "Accepted" if doc.accepted else f"Rejected ({doc.rejection_reason})"
            url_str = f" — [{doc.url}]({doc.url})" if doc.url else ""
            lines.append(
                f"- **[{doc.doc_id}]** {doc.source_type.upper()}: {doc.title}{url_str}  \n"
                f"  *{status}* | Credibility: {doc.credibility.value}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_footer(packet: EvidencePacket) -> str:
        return (
            f"## Report Metadata\n\n"
            f"- Signal ID: `{packet.signal_id}`\n"
            f"- Generated: {packet.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"- Evidence grade: {packet.evidence_grade.value if packet.evidence_grade else 'pending'}\n"
            f"- Human review required: {'Yes' if packet.human_review_required else 'No'}\n"
            f"- Report auto-generated: Yes — NOT for clinical or regulatory use without analyst review\n"
        )
