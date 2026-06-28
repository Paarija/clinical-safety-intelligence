"""
app/streamlit_app.py

Clinical Safety Intelligence Dashboard — 7-page Streamlit application.

Pages:
  1. Overview          — system description, limitation banner, quick-stats
  2. Data Quality      — missingness, dedup report, mapping confidence
  3. Signal Detection  — ranked signal table, ROR forest plot, trend chart
  4. Clinical Trials   — trial comparison table, arm rates, risk difference
  5. Evidence Review   — per-signal evidence cards, grade badge
  6. Case Study        — data-driven Markdown case studies and built-in example
  7. Evaluation        — benchmark table, top-K recovery

Run:
    streamlit run src/clinical_safety/app/streamlit_app.py

Important:
  - Every page shows the limitation_banner() BEFORE any data.
  - All data is loaded from processed pipeline outputs (parquet / JSON).
  - If pipeline has not been run, a clear placeholder is shown — no fake data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from clinical_safety.app.components import (
    data_unavailable,
    grade_badge,
    limitation_banner,
    metric_card,
    pipeline_not_run_warning,
    ror_forest_plot,
    signal_table,
)
from clinical_safety.common.paths import Paths
from clinical_safety.common.mock_data import is_mock_pipeline_data

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  :root {
      color-scheme: dark;
  }

  [data-testid="stAppViewContainer"] {
      background: #0f172a;
      color: #e5e7eb;
  }

  [data-testid="stSidebar"] {
      background: #1e293b;
  }

  [data-testid="stSidebar"] * {
      color: #e5e7eb !important;
  }

  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] .stCaption {
      color: #cbd5e1 !important;
  }

  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
      color: #e5e7eb !important;
  }

  [data-testid="stSidebar"] button {
      background: #f8fafc !important;
      color: #0f172a !important;
      border: 1px solid #cbd5e1 !important;
      font-weight: 600;
  }

  [data-testid="stSidebar"] [role="radiogroup"] label {
      min-height: 1.7rem;
      opacity: 1 !important;
  }

  [data-testid="metric-container"] {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 12px;
  }

  [data-testid="metric-container"] * {
      color: #e5e7eb !important;
  }

  h1, h2, h3, h4, h5, h6 {
      color: #f8fafc;
      letter-spacing: 0;
  }

  p, li, span, label, div {
      color: inherit;
  }

  [data-testid="stDataFrame"] { background: #1e293b; }
  .stAlert { border-radius: 8px; }

  .streamlit-expanderHeader {
      background: #1e293b !important;
      border-radius: 8px;
      color: #e5e7eb !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_paths() -> Paths:
    return Paths()


PATHS = get_paths()
CANDIDATE_SIGNALS_FILE = PATHS.processed_signals / "candidate_signals.parquet"
TRIAL_COMPARISON_FILE = PATHS.processed_analytics / "trial_comparison.parquet"
QUALITY_REPORT_FILE = PATHS.interim_quality / "data_quality_report.json"
EVIDENCE_PACKETS_DIR = PATHS.processed_evidence

CASE_STUDY_SOURCE_DIRS = (
    ("Curated case study", PATHS.case_studies),
    ("Generated report", PATHS.processed_reports),
)
CACHE_REFRESHED_AT_KEY = "clinical_safety_data_cache_refreshed_at"
CACHE_SIGNATURE_KEY = "clinical_safety_data_cache_signature"
CANDIDATE_SIGNAL_COLUMNS = {
    "drug_id",
    "event_id",
    "case_count",
    "ror",
    "ror_lower_ci",
    "ror_upper_ci",
}
TRIAL_COMPARISON_COLUMNS = {
    "nct_id",
    "drug_id",
    "event_id",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def cache_refreshed_at() -> datetime:
    if CACHE_REFRESHED_AT_KEY not in st.session_state:
        st.session_state[CACHE_REFRESHED_AT_KEY] = now_utc()
    return st.session_state[CACHE_REFRESHED_AT_KEY]


def artifact_signature(path: Path) -> tuple[str, float | None, int | None]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), stat.st_mtime, stat.st_size)


def directory_signature(
    path: Path,
    pattern: str = "*.json",
) -> tuple[tuple[str, float | None, int | None], ...]:
    if not path.exists():
        return ((str(path), None, None),)

    signatures = []
    for child in sorted(path.rglob(pattern)):
        if child.is_file():
            signatures.append(artifact_signature(child))
    return tuple(signatures)


def primary_artifacts_signature() -> tuple[tuple[str, float | None, int | None], ...]:
    return (
        artifact_signature(CANDIDATE_SIGNALS_FILE),
        artifact_signature(TRIAL_COMPARISON_FILE),
        artifact_signature(QUALITY_REPORT_FILE),
        artifact_signature(PATHS.source_manifest),
    )


def generated_artifacts_signature() -> tuple[tuple[str, float | None, int | None], ...]:
    return (
        *primary_artifacts_signature(),
        *directory_signature(EVIDENCE_PACKETS_DIR),
    )


def mock_data_signature() -> tuple[Any, ...]:
    return (
        artifact_signature(PATHS.processed_analytics / "_IS_MOCK_DATA"),
        *primary_artifacts_signature(),
        *directory_signature(PATHS.processed_evidence, pattern="*"),
        *directory_signature(PATHS.processed_reports, pattern="*"),
    )


def record_cache_signature(signature: tuple[tuple[str, float | None, int | None], ...]) -> None:
    if st.session_state.get(CACHE_SIGNATURE_KEY) != signature:
        st.session_state[CACHE_SIGNATURE_KEY] = signature
        st.session_state[CACHE_REFRESHED_AT_KEY] = now_utc()


def missing_columns(df: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(required_columns.difference(df.columns))


def show_missing_columns_message(name: str, missing: list[str]) -> bool:
    if not missing:
        return False

    data_unavailable(
        f"{name} artifact is missing required columns: {', '.join(missing)}. "
        "Re-run the pipeline step, then use Refresh Data Cache."
    )
    return True


def clear_data_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    signature = generated_artifacts_signature()
    st.session_state[CACHE_SIGNATURE_KEY] = signature
    st.session_state[CACHE_REFRESHED_AT_KEY] = now_utc()

def show_missing_file_artifact(path: Path, step: str) -> bool:
    """Show consistent guidance when an expected pipeline output file is absent."""
    if path.exists():
        return False

    pipeline_not_run_warning(step)
    data_unavailable(
        f"Expected pipeline artifact is missing: `{path}`. "
        "Run the relevant pipeline step, then use **Refresh Data Cache**."
    )
    return True


def show_empty_artifact_message(name: str) -> None:
    """Explain that an artifact exists but has no usable records."""
    data_unavailable(
        f"{name} artifact is present but contains no records. "
        "Re-run the pipeline step or inspect the upstream inputs."
    )


def evidence_packet_files_exist() -> bool:
    """Return True when at least one evidence packet JSON artifact is present."""
    return EVIDENCE_PACKETS_DIR.exists() and any(EVIDENCE_PACKETS_DIR.rglob("*.json"))


def show_missing_evidence_packets() -> bool:
    """Show consistent guidance when evidence packet artifacts are absent."""
    if evidence_packet_files_exist():
        return False

    pipeline_not_run_warning("LangGraph evidence workflow")
    if EVIDENCE_PACKETS_DIR.exists():
        detail = f"No evidence packet JSON files found under `{EVIDENCE_PACKETS_DIR}`."
    else:
        detail = f"Expected evidence packet directory is missing: `{EVIDENCE_PACKETS_DIR}`."
    data_unavailable(f"{detail} Run the evidence workflow, then use **Refresh Data Cache**.")
    return True


MOCK_DATA_WARNING = (
    "MOCK DATA - Synthetic evaluation pipeline outputs are loaded. "
    "Do not interpret displayed signals, trial comparisons, or source manifest counts as real safety findings."
)


@st.cache_data(ttl=300)
def is_mock_pipeline_data_cached(signature: tuple[Any, ...]) -> bool:
    return is_mock_pipeline_data(PATHS)


def mock_data_warning() -> None:
    if is_mock_pipeline_data_cached(mock_data_signature()):
        st.warning(MOCK_DATA_WARNING, icon=None)


def artifact_status(path: Path) -> str:
    signature = artifact_signature(path)
    if signature[1] is None:
        return "missing"
    modified = datetime.fromtimestamp(signature[1], tz=timezone.utc)
    return f"updated {format_utc(modified)}"


def directory_artifact_status(path: Path) -> str:
    if not path.exists():
        return "missing"

    files = [child for child in path.rglob("*.json") if child.is_file()]
    if not files:
        return "no JSON artifacts"

    newest = max(child.stat().st_mtime for child in files)
    modified = datetime.fromtimestamp(newest, tz=timezone.utc)
    return f"{len(files)} JSON artifact(s), newest updated {format_utc(modified)}"


def show_page_freshness(
    artifacts: dict[str, Path] | None = None,
    directories: dict[str, Path] | None = None,
) -> None:
    artifact_items = artifacts or {}
    directory_items = directories or {}
    statuses = [f"{label}: {artifact_status(path)}" for label, path in artifact_items.items()]
    statuses.extend(
        f"{label}: {directory_artifact_status(path)}" for label, path in directory_items.items()
    )
    if not statuses:
        return

    st.caption(
        "Data freshness — "
        + " | ".join(statuses)
        + f" | cache loaded/refreshed {format_utc(cache_refreshed_at())}"
    )


def data_freshness_panel() -> None:
    artifacts = {
        "Signals": CANDIDATE_SIGNALS_FILE,
        "Trial comparison": TRIAL_COMPARISON_FILE,
        "Quality report": QUALITY_REPORT_FILE,
        "Source manifest": PATHS.source_manifest,
    }
    record_cache_signature(generated_artifacts_signature())

    st.sidebar.subheader("Pipeline data")
    st.sidebar.caption(f"Cache loaded/refreshed: {format_utc(cache_refreshed_at())}")
    for label, path in artifacts.items():
        st.sidebar.caption(f"{label}: {artifact_status(path)}")
    st.sidebar.caption(f"Evidence packets: {directory_artifact_status(EVIDENCE_PACKETS_DIR)}")
    if st.sidebar.button("Refresh Data Cache", help="Clear cached artifact reads and reload from disk."):
        clear_data_caches()
        st.rerun()
# ── Data loaders (cached, load once per session) ──────────────────────────────

@st.cache_data(ttl=300)
def _load_candidate_signals(signature: tuple[str, float | None, int | None]) -> pd.DataFrame:
    if signature[1] is None:
        return pd.DataFrame()
    return pd.read_parquet(CANDIDATE_SIGNALS_FILE)


def load_candidate_signals() -> pd.DataFrame:
    return _load_candidate_signals(artifact_signature(CANDIDATE_SIGNALS_FILE))


@st.cache_data(ttl=300)
def _load_data_quality_report(signature: tuple[str, float | None, int | None]) -> dict[str, Any]:
    if signature[1] is None:
        return {}
    with QUALITY_REPORT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data_quality_report() -> dict[str, Any]:
    return _load_data_quality_report(artifact_signature(QUALITY_REPORT_FILE))


@st.cache_data(ttl=300)
def _load_trial_comparison(signature: tuple[str, float | None, int | None]) -> pd.DataFrame:
    if signature[1] is None:
        return pd.DataFrame()
    return pd.read_parquet(TRIAL_COMPARISON_FILE)


def load_trial_comparison() -> pd.DataFrame:
    return _load_trial_comparison(artifact_signature(TRIAL_COMPARISON_FILE))


@st.cache_data(ttl=300)
def _load_evidence_packets(signature: tuple[tuple[str, float | None, int | None], ...]) -> dict[str, Any]:
    """Load all saved EvidencePacket JSON files from processed/evidence_packets/."""
    packets: dict[str, Any] = {}
    if signature == ((str(EVIDENCE_PACKETS_DIR), None, None),):
        return packets
    for json_file in EVIDENCE_PACKETS_DIR.rglob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        signal_id = data.get("signal_id", json_file.stem)
        packets[signal_id] = data
    return packets


def load_evidence_packets() -> dict[str, Any]:
    return _load_evidence_packets(directory_signature(EVIDENCE_PACKETS_DIR))


@st.cache_data(ttl=300)
def _load_source_manifest(signature: tuple[str, float | None, int | None]) -> list[dict]:
    if signature[1] is None:
        return []
    with PATHS.source_manifest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        entries = data.get("entries", [])
    elif isinstance(data, list):
        entries = data
    else:
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def load_source_manifest() -> list[dict]:
    return _load_source_manifest(artifact_signature(PATHS.source_manifest))


def _relative_display_path(path: Path) -> str:
    """Return a stable, readable path for dashboard guidance."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _case_study_title(path: Path) -> str:
    """Create a readable selector label from a Markdown report filename."""
    stem = path.stem.removesuffix("_report")
    return stem.replace("__", " + ").replace("_", " ").replace("-", " ").title()


def discover_case_study_markdown() -> list[dict[str, Any]]:
    """Find curated or generated Markdown case-study/report files."""
    case_studies: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source_label, directory in CASE_STUDY_SOURCE_DIRS:
        if not directory.exists():
            continue
        markdown_files = sorted(
            path
            for pattern in ("*.md", "*.markdown")
            for path in directory.rglob(pattern)
            if path.is_file()
        )
        for markdown_file in markdown_files:
            resolved = markdown_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            case_studies.append(
                {
                    "label": source_label,
                    "path": resolved,
                    "title": _case_study_title(markdown_file),
                }
            )
    return case_studies


@st.cache_data(ttl=300)
def _load_case_study_markdown(path: str, signature: tuple[str, float | None, int | None]) -> str:
    if signature[1] is None:
        return ""
    return Path(path).read_text(encoding="utf-8")


def load_case_study_markdown(path: Path) -> str:
    return _load_case_study_markdown(str(path), artifact_signature(path))

# ── Sidebar navigation ────────────────────────────────────────────────────────

def sidebar_nav() -> str:
    """Render sidebar and return selected page name."""
    st.sidebar.markdown(
        "<div style='font-size:2rem; text-align:center; font-weight:700;'>CSI</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.title("Clinical Safety Intelligence")
    st.sidebar.caption("Research demo | GLP-1 agonists | V1")
    st.sidebar.divider()

    pages = [
        "Overview",
        "Data Quality",
        "Signal Detection",
        "Clinical Trials",
        "Evidence Review",
        "Case Study",
        "Evaluation",
    ]
    selection = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")
    data_freshness_panel()
    st.sidebar.divider()
    st.sidebar.caption(
        "Not for clinical use. Research and portfolio demonstration only."
    )
    return selection


# ── Page 1: Overview ──────────────────────────────────────────────────────────

def page_overview() -> None:
    st.title("Clinical Safety Intelligence")
    st.subheader("Signal Triage · Evidence Grading · Traceable Adverse-Event Review")
    limitation_banner()
    mock_data_warning()
    show_page_freshness({
        "Candidate signals": CANDIDATE_SIGNALS_FILE,
        "Trial comparison": TRIAL_COMPARISON_FILE,
        "Source manifest": PATHS.source_manifest,
    })
    st.markdown("""
    This system connects **FDA FAERS post-market adverse-event reports**, 
    **ClinicalTrials.gov results**, **FDA regulatory communications**, and 
    **PubMed literature** into analyst-ready safety signal reports for 
    **GLP-1 receptor agonists** (semaglutide, liraglutide, dulaglutide, tirzepatide, exenatide).
    """)

    # Quick stats
    signals_missing = show_missing_file_artifact(
        CANDIDATE_SIGNALS_FILE, "Signal detection pipeline"
    )
    trial_missing = show_missing_file_artifact(
        TRIAL_COMPARISON_FILE,
        "ClinicalTrials acquisition (ClinicalTrialsSource + TrialComparator)",
    )
    manifest_missing = show_missing_file_artifact(PATHS.source_manifest, "FAERS source acquisition")
    signals_df = pd.DataFrame() if signals_missing else load_candidate_signals()
    trial_df = pd.DataFrame() if trial_missing else load_trial_comparison()
    manifest = [] if manifest_missing else load_source_manifest()
    if not signals_missing and signals_df.empty:
        show_empty_artifact_message("Candidate signals")
    if not trial_missing and trial_df.empty:
        show_empty_artifact_message("Trial comparison")
    if not manifest_missing and not manifest:
        show_empty_artifact_message("Source manifest")
    if not trial_missing and not trial_df.empty:
        if show_missing_columns_message("Trial comparison", missing_columns(trial_df, {"nct_id"})):
            trial_df = pd.DataFrame()
    st.subheader("Pipeline Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Candidate Signals",
            len(signals_df) if not signals_df.empty else "—",
            help_text="Drug-event pairs passing signal detection thresholds",
        )
    with c2:
        faers_entries = sum(1 for e in manifest if e.get("source_type") == "faers")
        metric_card(
            "FAERS Source Files",
            faers_entries if faers_entries else "—",
            help_text="Raw FAERS files registered in source manifest",
        )
    with c3:
        trial_studies = trial_df["nct_id"].nunique() if not trial_df.empty else "—"
        metric_card(
            "Trial Studies",
            trial_studies,
            help_text="ClinicalTrials.gov studies with AE results",
        )
    with c4:
        grade_a = (
            len(signals_df[signals_df.get("evidence_grade", pd.Series()) == "A"])
            if not signals_df.empty and "evidence_grade" in signals_df.columns
            else "—"
        )
        metric_card(
            "Grade A Signals",
            grade_a,
            help_text="Signals with strongest evidence (Grade A) requiring analyst review",
        )

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    ```
    Raw Sources (FAERS ZIP + ClinicalTrials API + FDA API + PubMed)
        │
        ▼
    Acquisition → Parsing & Normalization → Signal Detection (ROR/PRR/CI)
        │
        ▼
    LangGraph Evidence Workflow:
      [FAERS Builder] → [Trial Builder] → [Regulatory] → [Literature]
      → [Quality Gate] → [Synthesizer (Gemini)] → [Grader A/B/C/D]
      → [Human Review?] → [Report Generator]
        │
        ▼
    This Dashboard + Markdown Reports
    ```
    """)

    st.divider()
    st.subheader("Evidence Grading")
    grade_rows = [
        {"Grade": "A", "Meaning": "Strong — ≥10 cases, ROR CI ≥2.0, serious outcomes, regulatory support", "Action": "High-priority analyst review"},
        {"Grade": "B", "Meaning": "Moderate — ≥5 cases, ROR CI ≥1.0", "Action": "Monitor and review additional evidence"},
        {"Grade": "C", "Meaning": "Weak — ≥3 cases, low ROR or contradictory evidence", "Action": "Keep on watchlist"},
        {"Grade": "D", "Meaning": "Insufficient — below thresholds, poor mapping", "Action": "Exploratory only"},
    ]
    st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)


# ── Page 2: Data Quality ──────────────────────────────────────────────────────

def page_data_quality() -> None:
    st.title("Data Quality")
    limitation_banner()
    mock_data_warning()
    show_page_freshness({"Quality report": QUALITY_REPORT_FILE})

    if show_missing_file_artifact(
        QUALITY_REPORT_FILE, "FAERS data ingestion and quality reporting"
    ):
        return
    report = load_data_quality_report()
    if not report:
        show_empty_artifact_message("Data quality report")
        return

    # Deduplication summary
    dedup = report.get("deduplication")
    if dedup:
        st.subheader("FAERS Case Deduplication")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total rows before dedup", dedup.get("total_rows_before", "—"))
        with c2:
            metric_card("Unique cases after dedup", dedup.get("unique_cases_after", "—"))
        with c3:
            rate = dedup.get("duplicate_rate_pct", None)
            metric_card("Duplicate rate", f"{rate:.1f}%" if rate is not None else "—")

    st.divider()

    # Per-table quality
    tables = report.get("tables", {})
    if tables:
        st.subheader("Table Schema & Missingness")
        for tname, tdata in tables.items():
            with st.expander(f"**{tname}** — {tdata.get('row_count', 0):,} rows", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Row Count", f"{tdata.get('row_count', 0):,}")
                    violations = tdata.get("schema_violations", [])
                    if violations:
                        st.error(f"Schema violations: {violations}")
                    else:
                        st.success("No schema violations")
                with c2:
                    high_miss = tdata.get("high_missingness_cols", [])
                    if high_miss:
                        st.warning(f"High missingness (>20%): {high_miss}")
                    miss = tdata.get("missingness_pct", {})
                    if miss:
                        miss_df = pd.DataFrame(
                            list(miss.items()), columns=["Column", "Missingness %"]
                        ).sort_values("Missingness %", ascending=False)
                        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    # Drug / event mapping
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        drug_conf = report.get("drug_mapping")
        if drug_conf:
            st.subheader("Drug Mapping Confidence")
            drug_df = pd.DataFrame(list(drug_conf.items()), columns=["Confidence", "Count"])
            try:
                import plotly.express as px
                fig = px.pie(drug_df, names="Confidence", values="Count",
                             title="Drug Name Mapping Confidence Distribution",
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(drug_df)
        else:
            data_unavailable("Drug mapping audit not available.")

    with col2:
        event_conf = report.get("event_mapping")
        if event_conf:
            st.subheader("Event Mapping Confidence")
            event_df = pd.DataFrame(list(event_conf.items()), columns=["Confidence", "Count"])
            try:
                import plotly.express as px
                fig = px.pie(event_df, names="Confidence", values="Count",
                             title="Event Term Mapping Confidence Distribution",
                             color_discrete_sequence=px.colors.sequential.Greens_r)
                fig.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(event_df)
        else:
            data_unavailable("Event mapping audit not available.")


# ── Page 3: Signal Detection ──────────────────────────────────────────────────

def page_signal_detection() -> None:
    st.title("Signal Detection")
    limitation_banner()
    mock_data_warning()
    show_page_freshness({"Candidate signals": CANDIDATE_SIGNALS_FILE})

    if show_missing_file_artifact(CANDIDATE_SIGNALS_FILE, "Signal detection (signal_ranking.py)"):
        return
    signals_df = load_candidate_signals()
    if signals_df.empty:
        show_empty_artifact_message("Candidate signals")
        return
    if show_missing_columns_message(
        "Candidate signals", missing_columns(signals_df, CANDIDATE_SIGNAL_COLUMNS)
    ):
        return

    st.markdown(f"**{len(signals_df)} candidate signals** passed detection thresholds.")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        drugs = ["All"] + sorted(signals_df["drug_id"].unique().tolist())
        drug_filter = st.selectbox("Filter by Drug", drugs, key="sd_drug")
    with col2:
        events = ["All"] + sorted(signals_df["event_id"].unique().tolist())
        event_filter = st.selectbox("Filter by Event", events, key="sd_event")
    with col3:
        min_cases = st.slider("Minimum Case Count", 1, int(signals_df["case_count"].max()), 3, key="sd_min_cases")

    filtered = signals_df.copy()
    if drug_filter != "All":
        filtered = filtered[filtered["drug_id"] == drug_filter]
    if event_filter != "All":
        filtered = filtered[filtered["event_id"] == event_filter]
    filtered = filtered[filtered["case_count"] >= min_cases]

    st.divider()
    st.subheader("Ranked Signal Table")
    signal_table(filtered)

    st.divider()
    st.subheader("ROR Forest Plot")
    ror_forest_plot(filtered)

    # Quick top-3 summary
    if not filtered.empty:
        st.divider()
        st.subheader("Top 3 Signals")
        for _, row in filtered.head(3).iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"**{row['drug_id'].title()} + {row['event_id'].replace('_', ' ').title()}**")
                with c2:
                    st.markdown(f"Cases: **{row['case_count']}**")
                with c3:
                    st.markdown(f"ROR: **{row['ror']:.2f}** (CI: {row['ror_lower_ci']:.2f}–{row['ror_upper_ci']:.2f})")
                with c4:
                    trial_avail = "Available" if row.get("trial_evidence_available") else "Unavailable"
                    st.markdown(f"Trial data: {trial_avail}")


# ── Page 4: Clinical Trials ───────────────────────────────────────────────────

def page_clinical_trials() -> None:
    st.title("Clinical Trial Comparison")
    limitation_banner()
    mock_data_warning()
    show_page_freshness({"Trial comparison": TRIAL_COMPARISON_FILE})

    if show_missing_file_artifact(
        TRIAL_COMPARISON_FILE,
        "ClinicalTrials acquisition (ClinicalTrialsSource + TrialComparator)",
    ):
        return
    trial_df = load_trial_comparison()
    if trial_df.empty:
        show_empty_artifact_message("Trial comparison")
        return
    if show_missing_columns_message(
        "Trial comparison", missing_columns(trial_df, TRIAL_COMPARISON_COLUMNS)
    ):
        return

    st.markdown(f"**{trial_df['nct_id'].nunique()} clinical trials** with AE data loaded.")

    # Drug+event selector
    drug_event_pairs = (
        trial_df.groupby(["drug_id", "event_id"]).size().reset_index()
    )
    drug_event_pairs["label"] = drug_event_pairs["drug_id"] + " + " + drug_event_pairs["event_id"]
    selected_label = st.selectbox(
        "Select Drug + Event Signal", drug_event_pairs["label"].tolist(), key="ct_pair"
    )
    selected_row = drug_event_pairs[drug_event_pairs["label"] == selected_label].iloc[0]
    drug_id = selected_row["drug_id"]
    event_id = selected_row["event_id"]

    subset = trial_df[(trial_df["drug_id"] == drug_id) & (trial_df["event_id"] == event_id)]

    st.divider()
    st.subheader(f"Trial Adverse Event Rates: {drug_id.title()} + {event_id.replace('_', ' ').title()}")
    st.dataframe(
        subset[[c for c in [
            "nct_id", "arm_label", "arm_type", "event_term",
            "is_serious", "affected", "at_risk", "event_rate",
            "comparator_rate", "absolute_risk_difference"
        ] if c in subset.columns]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    # Show risk difference where available
    if "absolute_risk_difference" not in subset.columns:
        data_unavailable("No arm-level risk difference data available for this signal.")
    else:
        ard_df = subset.dropna(subset=["absolute_risk_difference"])
        if not ard_df.empty:
            st.subheader("Absolute Risk Difference (Treatment − Comparator Rate)")
            try:
                import plotly.express as px
                summary = (
                    ard_df.groupby(["nct_id", "event_id"])["absolute_risk_difference"]
                    .first().reset_index()
                )
                fig = px.bar(
                    summary, x="event_id", y="absolute_risk_difference",
                    color="nct_id", barmode="group",
                    title="ARD by Study (positive = higher rate in treatment arm)",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
                fig.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(ard_df[["nct_id", "event_id", "absolute_risk_difference"]])
        else:
            data_unavailable("No arm-level risk difference data available for this signal.")


# ── Page 5: Evidence Review ───────────────────────────────────────────────────

def page_evidence_review() -> None:
    st.title("Evidence Review")
    limitation_banner()
    mock_data_warning()
    show_page_freshness(directories={"Evidence packets": EVIDENCE_PACKETS_DIR})

    if show_missing_evidence_packets():
        st.markdown("""
        **To generate evidence packets and reports, run:**
        ```bash
        python3 run_evidence_workflow.py --dry-run
        ```

        Use `python3 run_evidence_workflow.py` for real Gemini synthesis after `GEMINI_API_KEY`
        is set in `.env`.
        """)
        return

    packets = load_evidence_packets()
    if not packets:
        data_unavailable("Evidence packet artifacts were found, but no packet JSON could be loaded.")
        return

    signal_ids = list(packets.keys())
    selected_id = st.selectbox("Select Signal", signal_ids, key="ev_signal")
    packet_data = packets[selected_id]

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        grade = packet_data.get("evidence_grade", "?")
        st.markdown("### Evidence Grade")
        grade_badge(grade)
        st.markdown(f"**Triage:** {packet_data.get('triage_status', 'pending').replace('_', ' ').title()}")
        st.markdown(f"**Signal ID:** `{packet_data.get('signal_id', '—')}`")
        st.markdown(f"**Human Review Required:** {'Yes - analyst review required' if packet_data.get('human_review_required') else 'No'}")

    with col2:
        st.markdown("### Synthesis Summary")
        synthesis = packet_data.get("synthesis_summary")
        if synthesis:
            st.markdown(synthesis)
        else:
            data_unavailable("Synthesis not available (dry-run mode or no evidence passed quality gate).")

        if packet_data.get("synthesis_supports"):
            st.markdown("**Supporting Evidence:**")
            for s in packet_data["synthesis_supports"]:
                st.markdown(f"- {s}")

        if packet_data.get("synthesis_contradictions"):
            st.markdown("**Contradictions:**")
            for c in packet_data["synthesis_contradictions"]:
                st.markdown(f"- {c}")

    st.divider()

    # Evidence documents
    all_docs_data = (
        packet_data.get("regulatory_documents", [])
        + packet_data.get("literature_documents", [])
        + packet_data.get("label_documents", [])
    )

    if all_docs_data:
        st.subheader(f"Retrieved Evidence ({len(all_docs_data)} documents)")

        tab_reg, tab_lit, tab_label = st.tabs(["Regulatory", "Literature", "FDA Label"])
        with tab_reg:
            reg_docs = packet_data.get("regulatory_documents", [])
            for d in reg_docs:
                _render_doc_card(d)
            if not reg_docs:
                data_unavailable("No regulatory documents retrieved.")
        with tab_lit:
            lit_docs = packet_data.get("literature_documents", [])
            for d in lit_docs:
                _render_doc_card(d)
            if not lit_docs:
                data_unavailable("No literature documents retrieved.")
        with tab_label:
            label_docs = packet_data.get("label_documents", [])
            for d in label_docs:
                _render_doc_card(d)
            if not label_docs:
                data_unavailable("No FDA label documents retrieved.")
    else:
        data_unavailable("No evidence documents found in this packet.")

    # Evidence gaps
    gaps = packet_data.get("evidence_gaps", [])
    if gaps:
        st.divider()
        st.subheader("Evidence Gaps")
        for g in gaps:
            st.warning(g)


def _render_doc_card(doc_data: dict) -> None:
    """Render a document dict (from JSON) as an evidence card."""
    accepted = doc_data.get("accepted", True)
    rejection = doc_data.get("rejection_reason", "")
    doc_id = doc_data.get("doc_id", "unknown")
    title = doc_data.get("title", "Untitled")
    url = doc_data.get("url")
    snippet = doc_data.get("snippet")

    status = "Accepted" if accepted else "Rejected"
    label = f"{status} [{doc_id}] {title[:80]}{'...' if len(title) > 80 else ''}"

    with st.expander(label, expanded=False):
        st.write(f"**Source:** {doc_data.get('source_type', 'unknown').upper()}")
        st.write(f"**Credibility:** {doc_data.get('credibility', 'unknown')}")
        st.write(f"**Status:** {'Accepted' if accepted else f'Rejected ({rejection})'}")
        if url:
            st.markdown(f"**URL:** [{url}]({url})")
        if snippet:
            st.markdown(f"**Snippet:**\n\n> {snippet[:600]}")


# ── Page 6: Case Study ────────────────────────────────────────────────────────

def _show_case_study_guidance() -> None:
    """Explain how analysts can add case-study Markdown without code changes."""
    curated_dir = _relative_display_path(PATHS.case_studies)
    generated_dir = _relative_display_path(PATHS.processed_reports)
    st.info(
        "No curated or generated Markdown case-study files were found. "
        f"Add a `.md` or `.markdown` file under `{curated_dir}` to make it selectable here, "
        f"or generate evidence reports under `{generated_dir}` with the evidence workflow."
    )
    st.caption(
        "Markdown files are rendered as-authored. Dry-run/mock reports are placeholders and "
        "must not be interpreted as real clinical evidence."
    )


def _render_builtin_case_study_example() -> None:
    """Render the bundled semaglutide walkthrough using only live artifacts."""
    st.divider()
    st.subheader("Built-in example: Semaglutide + pancreatitis")
    st.caption(
        "This built-in walkthrough is not a generated case study. It demonstrates the expected "
        "review structure and only shows findings when the corresponding pipeline artifacts exist."
    )

    packets = load_evidence_packets() if evidence_packet_files_exist() else {}
    signal_id = "semaglutide__pancreatitis"
    packet_data = packets.get(signal_id)

    st.markdown("**Step 1: FAERS signal detection**")
    if show_missing_file_artifact(CANDIDATE_SIGNALS_FILE, "Signal detection"):
        pass
    else:
        signals_df = load_candidate_signals()
        if signals_df.empty:
            show_empty_artifact_message("Candidate signals")
        elif show_missing_columns_message(
            "Candidate signals", missing_columns(signals_df, CANDIDATE_SIGNAL_COLUMNS)
        ):
            pass
        else:
            sem_pan = signals_df[
                (signals_df["drug_id"] == "semaglutide")
                & (signals_df["event_id"] == "pancreatitis")
            ]
            if not sem_pan.empty:
                row = sem_pan.iloc[0]
                seriousness_rate = row.get("seriousness_rate")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Case Count", int(row["case_count"]))
                with c2:
                    metric_card("ROR", f"{row['ror']:.3f}")
                with c3:
                    metric_card("ROR 95% CI Lower", f"{row['ror_lower_ci']:.3f}")
                with c4:
                    metric_card(
                        "Seriousness Rate",
                        f"{seriousness_rate:.1%}" if pd.notna(seriousness_rate) else "N/A",
                    )
                ror_forest_plot(sem_pan)
            else:
                data_unavailable(
                    "Semaglutide + pancreatitis was not found in ranked signals. "
                    "Run signal detection with data that includes this pair."
                )

    st.markdown("**Step 2: Clinical trial comparison**")
    if show_missing_file_artifact(TRIAL_COMPARISON_FILE, "ClinicalTrials acquisition"):
        pass
    else:
        trial_df = load_trial_comparison()
        if trial_df.empty:
            show_empty_artifact_message("Trial comparison")
        elif show_missing_columns_message(
            "Trial comparison", missing_columns(trial_df, {"drug_id", "event_id"})
        ):
            pass
        else:
            tc_subset = trial_df[
                (trial_df["drug_id"] == "semaglutide")
                & (trial_df["event_id"] == "pancreatitis")
            ]
            if not tc_subset.empty:
                st.dataframe(tc_subset, use_container_width=True, hide_index=True)
            else:
                data_unavailable("No trial AE data for semaglutide + pancreatitis.")

    st.markdown("**Step 3: Evidence synthesis and grade**")
    if packet_data:
        col1, col2 = st.columns([1, 3])
        with col1:
            grade_badge(packet_data.get("evidence_grade", "?"))
        with col2:
            st.markdown(packet_data.get("grade_explanation", "*No explanation available.*"))

        synthesis = packet_data.get("synthesis_summary")
        if synthesis:
            st.markdown("**Synthesis:**")
            st.markdown(f"> {synthesis}")
    elif show_missing_evidence_packets():
        pass
    else:
        data_unavailable(
            "Evidence packet for semaglutide + pancreatitis has not been generated yet. "
            "Run `python3 run_evidence_workflow.py` after setting GEMINI_API_KEY in `.env`, "
            "or use dry-run mode only for visibly placeholder output."
        )


def page_case_study() -> None:
    st.title("Case Studies")
    limitation_banner()
    mock_data_warning()

    st.markdown(
        "Curated case-study Markdown from `reports/case_studies/` and generated Markdown "
        "reports from processed outputs are discovered automatically."
    )

    case_studies = discover_case_study_markdown()
    if case_studies:
        selected_index = st.selectbox(
            "Case-study Markdown file",
            list(range(len(case_studies))),
            format_func=lambda index: (
                f"{case_studies[index]['title']} "
                f"({case_studies[index]['label']}: "
                f"{_relative_display_path(case_studies[index]['path'])})"
            ),
        )
        selected = case_studies[selected_index]
        selected_path = selected["path"]
        st.caption(
            f"Source: {selected['label']} · `{_relative_display_path(selected_path)}`"
        )
        if selected["label"] == "Generated report":
            st.info(
                "This file was produced by the pipeline report generator. Confirm whether "
                "the underlying run used real external evidence or dry-run placeholders "
                "before citing it."
            )
        content = load_case_study_markdown(selected_path).strip()
        if content:
            st.markdown(content)
        else:
            data_unavailable(
                f"Selected Markdown file is empty: `{_relative_display_path(selected_path)}`."
            )
        return

    _show_case_study_guidance()
    _render_builtin_case_study_example()


# ── Page 7: Evaluation ────────────────────────────────────────────────────────

def page_evaluation() -> None:
    st.title("Evaluation - Known Signal Benchmark")
    limitation_banner()
    mock_data_warning()
    show_page_freshness({"Candidate signals": CANDIDATE_SIGNALS_FILE})

    st.markdown("""
    The benchmark tests whether the signal detection system recovers **known positive signals**
    (confirmed by FDA label warnings or regulatory action) and avoids ranking **weak controls**
    (no known GLP-1 association) highly.
    """)

    # Known positives
    known_positives = pd.DataFrame([
        {"Drug": "semaglutide", "Event": "pancreatitis", "Evidence Basis": "FDA label warning, SUSTAIN/STEP trials"},
        {"Drug": "liraglutide", "Event": "gallbladder disease", "Evidence Basis": "FDA label warning, SCALE trial"},
        {"Drug": "semaglutide", "Event": "gallbladder disease", "Evidence Basis": "FDA label warning"},
        {"Drug": "liraglutide", "Event": "pancreatitis", "Evidence Basis": "FDA label warning"},
        {"Drug": "exenatide", "Event": "pancreatitis", "Evidence Basis": "FDA label warning, class effect"},
        {"Drug": "dulaglutide", "Event": "pancreatitis", "Evidence Basis": "FDA label warning, class effect"},
    ])

    weak_controls = pd.DataFrame([
        {"Drug": "semaglutide", "Event": "intracranial hemorrhage", "Rationale": "No known association"},
        {"Drug": "liraglutide", "Event": "aplastic anemia", "Rationale": "No known association"},
        {"Drug": "exenatide", "Event": "parkinson's disease", "Rationale": "No established PV signal"},
    ])

    st.subheader("Known Positive Signals (should appear in top-ranked list)")
    st.dataframe(known_positives, use_container_width=True, hide_index=True)

    st.subheader("Weak Controls (should NOT rank highly)")
    st.dataframe(weak_controls, use_container_width=True, hide_index=True)

    st.divider()

    # Top-K recovery evaluation
    if show_missing_file_artifact(CANDIDATE_SIGNALS_FILE, "Signal detection pipeline"):
        return
    signals_df = load_candidate_signals()
    if signals_df.empty:
        show_empty_artifact_message("Candidate signals")
        return
    if show_missing_columns_message(
        "Candidate signals", missing_columns(signals_df, {"drug_id", "event_id"})
    ):
        return

    st.subheader("Top-K Recovery")
    top_n = st.slider("Evaluate top-K signals", 5, 50, 20, key="eval_k")
    top_k = signals_df.head(top_n)

    recovery_rows = []
    for _, kp in known_positives.iterrows():
        drug = kp["Drug"]
        event = kp["Event"].replace(" ", "_").replace("'s", "s")
        in_top_k = any(
            (row["drug_id"] == drug) and (row["event_id"] in event or event in row["event_id"])
            for _, row in top_k.iterrows()
        )
        rank = None
        match_rows = signals_df[
            (signals_df["drug_id"] == drug) &
            (signals_df["event_id"].str.contains(event.split("_")[0], case=False, na=False))
        ]
        if not match_rows.empty:
            rank = int(match_rows.index[0])
        recovery_rows.append({
            "Drug": drug,
            "Event": kp["Event"],
            f"In Top {top_n}": "Yes" if in_top_k else "No",
            "Rank": rank if rank else "Not ranked",
        })

    recovery_df = pd.DataFrame(recovery_rows)
    recovered = sum(1 for r in recovery_rows if r[f"In Top {top_n}"] == "Yes")
    total = len(recovery_rows)

    c1, c2 = st.columns(2)
    with c1:
        metric_card(f"Known Positives in Top {top_n}", f"{recovered}/{total}")
    with c2:
        precision = recovered / total if total else 0
        metric_card("Recovery Rate", f"{precision:.0%}")

    st.dataframe(recovery_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("False Positive Review")
    st.markdown(
        "*For each signal in the top K that is NOT a known positive, "
        "a qualitative review would assess: mapping errors, publicity effects, "
        "co-prescription confounders. This section is populated manually after pipeline run.*"
    )


# ── Main router ───────────────────────────────────────────────────────────────

def main() -> None:
    page = sidebar_nav()

    if "Overview" in page:
        page_overview()
    elif "Data Quality" in page:
        page_data_quality()
    elif "Signal Detection" in page:
        page_signal_detection()
    elif "Clinical Trials" in page:
        page_clinical_trials()
    elif "Evidence Review" in page:
        page_evidence_review()
    elif "Case Study" in page:
        page_case_study()
    elif "Evaluation" in page:
        page_evaluation()


if __name__ == "__main__":
    main()
