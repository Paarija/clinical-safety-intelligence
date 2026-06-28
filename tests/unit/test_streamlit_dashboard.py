from __future__ import annotations

import importlib
import json
import sys

import pandas as pd
import pytest


def test_dashboard_imports_and_loads_minimal_pipeline_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    signals_dir = data_dir / "processed" / "signals"
    analytics_dir = data_dir / "processed" / "analytics"
    quality_dir = data_dir / "interim" / "quality_reports"
    evidence_dir = data_dir / "processed" / "evidence_packets" / "semaglutide__pancreatitis"
    interim_dir = data_dir / "interim"

    for directory in (signals_dir, analytics_dir, quality_dir, evidence_dir, interim_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "drug_id": "semaglutide",
                "event_id": "pancreatitis",
                "case_count": 3,
                "ror": 2.4,
                "ror_lower_ci": 1.2,
                "ror_upper_ci": 4.8,
                "evidence_grade": "C",
            }
        ]
    ).to_parquet(signals_dir / "candidate_signals.parquet")
    pd.DataFrame(
        [
            {
                "nct_id": "NCT00000001",
                "drug_id": "semaglutide",
                "event_id": "pancreatitis",
            }
        ]
    ).to_parquet(analytics_dir / "trial_comparison.parquet")
    (quality_dir / "data_quality_report.json").write_text(
        json.dumps({"deduplication": {"unique_cases_after": 1}}),
        encoding="utf-8",
    )
    (interim_dir / "source_manifest.json").write_text(
        json.dumps({"entries": [{"source_type": "faers", "path": "sample.zip"}]}),
        encoding="utf-8",
    )
    (evidence_dir / "packet.json").write_text(
        json.dumps(
            {
                "signal_id": "semaglutide__pancreatitis",
                "evidence_grade": "C",
                "triage_status": "monitor",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    sys.modules.pop("clinical_safety.app.streamlit_app", None)
    dashboard = importlib.import_module("clinical_safety.app.streamlit_app")
    dashboard.clear_data_caches()

    signals = dashboard.load_candidate_signals()
    trial_comparison = dashboard.load_trial_comparison()

    assert dashboard.missing_columns(signals, dashboard.CANDIDATE_SIGNAL_COLUMNS) == []
    assert dashboard.missing_columns(trial_comparison, dashboard.TRIAL_COMPARISON_COLUMNS) == []
    assert dashboard.load_data_quality_report()["deduplication"]["unique_cases_after"] == 1
    assert dashboard.load_source_manifest()[0]["source_type"] == "faers"
    assert dashboard.evidence_packet_files_exist()
    assert dashboard.load_evidence_packets()["semaglutide__pancreatitis"]["evidence_grade"] == "C"
