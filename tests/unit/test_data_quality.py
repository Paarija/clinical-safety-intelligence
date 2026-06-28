"""
Unit tests for data quality reporting.
"""

from __future__ import annotations

import json

import pandas as pd

from clinical_safety.common.paths import Paths
from clinical_safety.common.types import MappingConfidence
from clinical_safety.quality.data_quality import DataQualityReporter


def test_run_includes_mapping_coverage_sections_when_summaries_passed(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    reporter = DataQualityReporter(paths=Paths(tmp_path))
    drug_summary = {
        MappingConfidence.EXACT.value: 1,
        MappingConfidence.ALIAS.value: 1,
        MappingConfidence.FUZZY.value: 0,
        MappingConfidence.UNMATCHED.value: 1,
    }
    event_summary = {
        MappingConfidence.EXACT.value: 9,
        MappingConfidence.ALIAS.value: 0,
        MappingConfidence.FUZZY.value: 0,
        MappingConfidence.UNMATCHED.value: 1,
    }

    report = reporter.run(
        tables={"DRUG": pd.DataFrame({"drugname": ["OZEMPIC"]})},
        drug_confidence_summary=drug_summary,
        event_confidence_summary=event_summary,
    )
    out = reporter.save(report)
    saved_report = json.loads(out.read_text(encoding="utf-8"))

    assert saved_report["drug_mapping_coverage"] == {
        "scope": "all_unique_terms_seen",
        "total": 3,
        "matched": 2,
        "unmatched": 1,
        "coverage_pct": 66.67,
    }
    assert saved_report["event_mapping_coverage"] == {
        "scope": "all_unique_terms_seen",
        "total": 10,
        "matched": 9,
        "unmatched": 1,
        "coverage_pct": 90.0,
    }
