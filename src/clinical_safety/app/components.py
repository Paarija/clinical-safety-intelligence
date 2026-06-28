"""
app/components.py

Reusable Streamlit UI components for the Clinical Safety Intelligence dashboard.

Components:
  - limitation_banner()         — persistent warning banner
  - grade_badge()               — coloured grade pill
  - signal_table()              — ranked signals table with ROR/CI
  - evidence_card()             — expandable evidence document card
  - ror_forest_plot()           — Plotly ROR + CI forest plot
  - trend_sparkline()           — Plotly quarterly trend line
  - metric_card()               — KPI card with label + value
  - data_unavailable()          — consistent placeholder for missing data

Usage:
    import streamlit as st
    from clinical_safety.app.components import limitation_banner, grade_badge
    limitation_banner()
    grade_badge("A")
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

# Grade → colour mapping (CSS hex)
_GRADE_COLORS = {
    "A": "#16a34a",   # green
    "B": "#ca8a04",   # yellow/amber
    "C": "#ea580c",   # orange
    "D": "#dc2626",   # red
}
_GRADE_LABELS = {
    "A": "Grade A — Strong Signal",
    "B": "Grade B — Moderate Signal",
    "C": "Grade C — Weak Signal",
    "D": "Grade D — Insufficient Evidence",
}


def limitation_banner() -> None:
    """
    Render a persistent limitations warning banner.

    IMPORTANT: This must appear on EVERY page of the dashboard.
    No chart or table should appear before this banner.
    """
    st.warning(
        "**Safety-Signal Triage.** "
        "FAERS reports cannot establish causality. ROR/PRR are *reporting-rate ratios*, "
        "not incidence estimates. Outputs require independent expert review before use "
        "in clinical, prescribing, regulatory, or operational safety decisions."
    )


def grade_badge(grade: str) -> None:
    """
    Render a coloured evidence grade badge.

    Args:
        grade: 'A', 'B', 'C', or 'D'
    """
    color = _GRADE_COLORS.get(grade.upper(), "#6b7280")
    label = _GRADE_LABELS.get(grade.upper(), f"Grade {grade} — Unknown")
    st.markdown(
        f"<span style='background:{color}; color:white; padding:4px 14px; "
        f"border-radius:12px; font-weight:bold; font-size:1rem;'>{label}</span>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, delta: str | None = None, help_text: str | None = None) -> None:
    """
    Render a KPI-style metric card using st.metric.

    Args:
        label    : Metric label.
        value    : Metric value (displayed prominently).
        delta    : Optional delta string (e.g. 'vs previous quarter').
        help_text: Optional tooltip.
    """
    st.metric(label=label, value=str(value), delta=delta, help=help_text)


def signal_table(df: pd.DataFrame, on_select: bool = False) -> pd.DataFrame | None:
    """
    Render the ranked signal table with key columns highlighted.

    Args:
        df       : Ranked signal DataFrame from candidate_signals.parquet.
        on_select: If True, return the selected row when user clicks.

    Returns:
        Selected row DataFrame (or None if not on_select).
    """
    if df.empty:
        data_unavailable("No signals found. Run the signal detection pipeline first.")
        return None

    display_cols = [c for c in [
        "rank", "drug_id", "event_id", "case_count",
        "ror", "ror_lower_ci", "ror_upper_ci",
        "prr", "seriousness_rate", "death_count",
        "trial_evidence_available", "rank_score",
    ] if c in df.columns]

    display_df = df[display_cols].copy()

    # Rename for readability
    rename = {
        "drug_id": "Drug", "event_id": "Event",
        "case_count": "Cases", "ror": "ROR",
        "ror_lower_ci": "ROR CI Lower", "ror_upper_ci": "ROR CI Upper",
        "prr": "PRR", "seriousness_rate": "Seriousness Rate",
        "death_count": "Deaths", "trial_evidence_available": "Trial Data",
        "rank_score": "Rank Score", "rank": "Rank",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename.items() if k in display_df.columns})

    # Format floats
    for col in ["ROR", "ROR CI Lower", "ROR CI Upper", "PRR", "Rank Score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    if "Seriousness Rate" in display_df.columns:
        display_df["Seriousness Rate"] = display_df["Seriousness Rate"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    return None


def ror_forest_plot(df: pd.DataFrame) -> None:
    """
    Render a Plotly ROR forest plot with confidence intervals.

    Args:
        df: DataFrame with drug_id, event_id, ror, ror_lower_ci, ror_upper_ci columns.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotly not installed — cannot render forest plot.")
        return

    if df.empty or "ror" not in df.columns:
        data_unavailable("No signal data available for forest plot.")
        return

    # Top 20 only
    plot_df = df.head(20).copy()
    plot_df["label"] = plot_df["drug_id"] + " + " + plot_df["event_id"]

    fig = go.Figure()

    # CI error bars
    fig.add_trace(go.Scatter(
        x=plot_df["ror"],
        y=plot_df["label"],
        mode="markers",
        marker=dict(size=10, color="#2563eb", symbol="square"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(plot_df["ror_upper_ci"] - plot_df["ror"]).tolist(),
            arrayminus=(plot_df["ror"] - plot_df["ror_lower_ci"]).tolist(),
            color="#2563eb",
            thickness=2,
        ),
        name="ROR (95% CI)",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ROR: %{x:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Null line at ROR = 1
    fig.add_vline(x=1, line_dash="dash", line_color="#9ca3af", annotation_text="ROR = 1")

    fig.update_layout(
        title="Reporting Odds Ratio (ROR) with 95% CI — Top Signals",
        xaxis_title="ROR (log scale)",
        yaxis_title="",
        xaxis_type="log",
        height=max(300, len(plot_df) * 30 + 100),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=200, r=40, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "ROR is a measure of *disproportionate reporting*, not drug incidence. "
        "A high ROR does not mean the event is common or caused by the drug."
    )


def trend_sparkline(trend_df: pd.DataFrame, drug_id: str, event_id: str) -> None:
    """
    Render a simple quarterly trend line for one drug-event pair.

    Args:
        trend_df : DataFrame with columns [quarter, drug_id, event_id, case_count].
        drug_id  : Drug to filter.
        event_id : Event to filter.
    """
    try:
        import plotly.express as px
    except ImportError:
        st.warning("plotly not installed — cannot render trend chart.")
        return

    if trend_df.empty:
        data_unavailable("No trend data available.")
        return

    pair_df = trend_df[
        (trend_df["drug_id"] == drug_id) & (trend_df["event_id"] == event_id)
    ].sort_values("quarter")

    if pair_df.empty:
        st.info(f"No quarterly trend data for {drug_id} + {event_id}.")
        return

    fig = px.line(
        pair_df,
        x="quarter",
        y="case_count",
        markers=True,
        title=f"Quarterly Report Count: {drug_id.title()} + {event_id.replace('_', ' ').title()}",
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)


def evidence_card(doc: Any) -> None:
    """
    Render an expandable evidence document card.

    Args:
        doc: EvidenceDocument object.
    """
    status = "Accepted" if doc.accepted else "Rejected"
    label = f"{status} [{doc.doc_id}] {doc.title[:80]}{'...' if len(doc.title) > 80 else ''}"

    with st.expander(label, expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Source type:** {doc.source_type.upper()}")
            st.write(f"**Credibility:** {doc.credibility.value}")
        with col2:
            st.write(f"**Status:** {'Accepted' if doc.accepted else f'Rejected ({doc.rejection_reason})'}")
            if doc.publication_date:
                st.write(f"**Published:** {doc.publication_date}")
        if doc.url:
            st.markdown(f"**URL:** [{doc.url}]({doc.url})")
        if doc.snippet:
            st.markdown(f"**Snippet:**\n\n> {doc.snippet[:600]}{'...' if len(doc.snippet) > 600 else ''}")


def data_unavailable(message: str = "Data not available.") -> None:
    """Render a consistent placeholder for missing pipeline data."""
    st.info(message)


def pipeline_not_run_warning(step: str) -> None:
    """Show a clear message when a pipeline step has not been run yet."""
    st.warning(
        f"**{step} has not been run yet.**  \n"
        "Run the data pipeline first, then refresh this page.  \n"
        "See the README for quickstart instructions."
    )
