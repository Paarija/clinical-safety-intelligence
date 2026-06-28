"""
Unit tests for the evaluation runner and metrics calculations.
"""

from __future__ import annotations

import pandas as pd

from clinical_safety.evaluation.runner import EvaluationRunner


def test_evaluate_dataframe_all_recovered():
    # Construct a DataFrame where all 6 known positives are in the top ranks
    signals = [
        {"drug_id": "semaglutide", "event_id": "pancreatitis", "rank_score": 10.0},
        {"drug_id": "liraglutide", "event_id": "gallbladder_disease", "rank_score": 9.0},
        {"drug_id": "semaglutide", "event_id": "gallbladder_disease", "rank_score": 8.0},
        {"drug_id": "liraglutide", "event_id": "pancreatitis", "rank_score": 7.0},
        {"drug_id": "exenatide", "event_id": "pancreatitis", "rank_score": 6.0},
        {"drug_id": "dulaglutide", "event_id": "pancreatitis", "rank_score": 5.0},
    ]
    df = pd.DataFrame(signals)
    results = EvaluationRunner.evaluate_dataframe(df)

    assert results["known_positives_total"] == 6
    assert results["recovered_positives_k10"] == 6
    assert results["recovered_positives_k20"] == 6
    assert results["recovery_rate_k10"] == 1.0
    assert results["recovery_rate_k20"] == 1.0
    assert results["weak_controls_in_top5"] == 0


def test_evaluate_dataframe_none_recovered():
    df = pd.DataFrame(columns=["drug_id", "event_id", "rank_score"])
    results = EvaluationRunner.evaluate_dataframe(df)

    assert results["recovered_positives_k10"] == 0
    assert results["recovery_rate_k10"] == 0.0


def test_evaluate_dataframe_weak_controls():
    # Put a weak control in the top 5
    signals = [
        {"drug_id": "semaglutide", "event_id": "intracranial_hemorrhage", "rank_score": 100.0},
    ]
    df = pd.DataFrame(signals)
    results = EvaluationRunner.evaluate_dataframe(df)

    assert results["weak_controls_in_top5"] == 1
