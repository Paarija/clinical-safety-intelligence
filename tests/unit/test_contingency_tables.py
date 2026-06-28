from __future__ import annotations

import pandas as pd

from clinical_safety.analytics.contingency_tables import ContingencyTableBuilder
from clinical_safety.common.paths import Paths


def test_contingency_tables_filter_and_carry_pair_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    drug_df = pd.DataFrame(
        [
            {
                "primaryid": 1,
                "drug_id": "semaglutide",
                "role_cod": "PS",
                "drug_mapping_confidence": "exact",
            },
            {
                "primaryid": 1,
                "drug_id": "semaglutide",
                "role_cod": "PS",
                "drug_mapping_confidence": "alias",
            },
            {
                "primaryid": 2,
                "drug_id": "semaglutide",
                "role_cod": "PS",
                "drug_mapping_confidence": "alias",
            },
            {
                "primaryid": 3,
                "drug_id": "semaglutide",
                "role_cod": "PS",
                "drug_mapping_confidence": "fuzzy",
            },
        ]
    )
    reac_df = pd.DataFrame(
        [
            {
                "primaryid": 1,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "exact",
            },
            {
                "primaryid": 2,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "alias",
            },
            {
                "primaryid": 3,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "fuzzy",
            },
        ]
    )

    result = ContingencyTableBuilder(paths=Paths()).build(drug_df, reac_df, save=False)

    row = result.iloc[0]
    assert row["a"] == 2
    assert row["drug_mapping_confidence"] == "alias"
    assert row["event_mapping_confidence"] == "alias"


def test_contingency_tables_sensitivity_includes_secondary_suspects(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    drug_df = pd.DataFrame(
        [
            {
                "primaryid": 1,
                "drug_id": "semaglutide",
                "role_cod": "PS",
                "drug_mapping_confidence": "exact",
            },
            {
                "primaryid": 2,
                "drug_id": "semaglutide",
                "role_cod": "SS",
                "drug_mapping_confidence": "exact",
            },
            {
                "primaryid": 3,
                "drug_id": "semaglutide",
                "role_cod": "C",
                "drug_mapping_confidence": "exact",
            },
        ]
    )
    reac_df = pd.DataFrame(
        [
            {
                "primaryid": 1,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "exact",
            },
            {
                "primaryid": 2,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "exact",
            },
            {
                "primaryid": 3,
                "event_id": "pancreatitis",
                "event_mapping_confidence": "exact",
            },
        ]
    )

    builder = ContingencyTableBuilder(paths=Paths())
    default = builder.build(drug_df, reac_df, save=False)
    sensitivity = builder.build(drug_df, reac_df, save=False, sensitivity=True)

    default_row = default.query(
        "drug_id == 'semaglutide' and event_id == 'pancreatitis'"
    ).iloc[0]
    sensitivity_row = sensitivity.query(
        "drug_id == 'semaglutide' and event_id == 'pancreatitis'"
    ).iloc[0]

    assert default_row["a"] == 1
    assert default_row["c"] == 2
    assert sensitivity_row["a"] == 2
    assert sensitivity_row["c"] == 1
