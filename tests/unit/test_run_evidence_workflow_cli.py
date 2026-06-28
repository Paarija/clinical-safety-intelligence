from __future__ import annotations

import json

import pandas as pd

import run_evidence_workflow as workflow_cli


def test_packet_has_real_synthesis_rejects_dry_run_packet(tmp_path) -> None:
    packet = tmp_path / "signal.json"
    packet.write_text(
        json.dumps(
            {
                "synthesis_summary": "Dry-run mode was used, so no Gemini synthesis was requested.",
                "evidence_grade": "C",
            }
        ),
        encoding="utf-8",
    )

    assert not workflow_cli._packet_has_real_synthesis(packet)


def test_packet_has_real_synthesis_accepts_real_packet(tmp_path) -> None:
    packet = tmp_path / "signal.json"
    packet.write_text(
        json.dumps(
            {
                "synthesis_summary": "Evidence was synthesized from retrieved sources.",
                "evidence_grade": "B",
            }
        ),
        encoding="utf-8",
    )

    assert workflow_cli._packet_has_real_synthesis(packet)


def test_row_to_metrics_preserves_signal_fields() -> None:
    row = pd.Series(
        {
            "drug_id": "semaglutide",
            "event_id": "pancreatitis",
            "evidence_window": "2026Q1",
            "case_count": 12,
            "ror": 2.5,
            "ror_lower_ci": 1.2,
            "ror_upper_ci": 4.1,
            "drug_mapping_confidence": "alias",
            "event_mapping_confidence": "exact",
        }
    )

    metrics = workflow_cli._row_to_metrics(row)

    assert metrics.drug_id == "semaglutide"
    assert metrics.event_id == "pancreatitis"
    assert metrics.case_count == 12
    assert metrics.ror_lower_ci == 1.2
    assert metrics.drug_mapping_confidence.value == "alias"
