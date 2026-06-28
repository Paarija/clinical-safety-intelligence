from __future__ import annotations

import json

import pytest

from clinical_safety.acquisition.clinicaltrials_source import ClinicalTrialsSource
from clinical_safety.common.exceptions import DataSourceError
from clinical_safety.modeling.trial_comparator import TrialComparator


class _ClinicalTrialsConfig:
    request_connect_timeout_sec = 1
    request_read_timeout_sec = 1


class _EventNormalizer:
    def normalize(self, term: str) -> tuple[str, str]:
        return f"event:{term.lower()}", term


def _valid_study() -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {
                "armGroups": [
                    {"label": "Placebo", "type": "PLACEBO_COMPARATOR"},
                ],
            },
        },
        "resultsSection": {
            "adverseEventsModule": {
                "eventGroups": [{"id": "EG1", "title": "Placebo"}],
                "seriousEvents": [
                    {
                        "term": "Nausea",
                        "stats": [{"groupId": "EG1", "numAffected": "1", "numAtRisk": "10"}],
                    },
                ],
                "otherEvents": [],
            },
        },
    }


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_get_page_rejects_non_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ClinicalTrialsSource.__new__(ClinicalTrialsSource)
    source._ct_cfg = _ClinicalTrialsConfig()

    def fake_get(*args: object, **kwargs: object) -> _Response:
        return _Response([])

    monkeypatch.setattr("clinical_safety.acquisition.clinicaltrials_source.requests.get", fake_get)

    with pytest.raises(DataSourceError, match="expected an object.*studies"):
        source._get_page({})


def test_get_page_rejects_non_list_studies(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ClinicalTrialsSource.__new__(ClinicalTrialsSource)
    source._ct_cfg = _ClinicalTrialsConfig()

    def fake_get(*args: object, **kwargs: object) -> _Response:
        return _Response({"studies": None})

    monkeypatch.setattr("clinical_safety.acquisition.clinicaltrials_source.requests.get", fake_get)

    with pytest.raises(DataSourceError, match="studies"):
        source._get_page({})


def test_get_page_rejects_study_without_nct_id(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ClinicalTrialsSource.__new__(ClinicalTrialsSource)
    source._ct_cfg = _ClinicalTrialsConfig()

    def fake_get(*args: object, **kwargs: object) -> _Response:
        return _Response({"studies": [{"protocolSection": {"identificationModule": {}}}]})

    monkeypatch.setattr("clinical_safety.acquisition.clinicaltrials_source.requests.get", fake_get)

    with pytest.raises(DataSourceError, match="nctId"):
        source._get_page({"query.intr": "semaglutide"})


def test_get_page_accepts_valid_study_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ClinicalTrialsSource.__new__(ClinicalTrialsSource)
    source._ct_cfg = _ClinicalTrialsConfig()

    def fake_get(*args: object, **kwargs: object) -> _Response:
        return _Response(
            {"studies": [_valid_study()], "nextPageToken": "next", "totalCount": 1}
        )

    monkeypatch.setattr("clinical_safety.acquisition.clinicaltrials_source.requests.get", fake_get)

    page = source._get_page({"query.intr": "semaglutide"})

    assert page["nextPageToken"] == "next"
    nct_id = page["studies"][0]["protocolSection"]["identificationModule"]["nctId"]
    assert nct_id == "NCT00000001"


def test_trial_comparator_rejects_raw_file_with_non_list_studies(tmp_path) -> None:
    raw_file = tmp_path / "drug-a.json"
    raw_file.write_text(json.dumps({"drug_id": "drug-a", "studies": None}), encoding="utf-8")

    comparator = TrialComparator.__new__(TrialComparator)

    with pytest.raises(DataSourceError, match="raw file drug-a.json schema mismatch"):
        comparator._process_drug_file(raw_file)


def test_trial_comparator_skips_malformed_study_with_context(tmp_path, caplog) -> None:
    raw_file = tmp_path / "drug-a.json"
    raw_file.write_text(
        json.dumps({
            "drug_id": "drug-a",
            "studies": [{"resultsSection": {}}, _valid_study()],
        }),
        encoding="utf-8",
    )

    comparator = TrialComparator.__new__(TrialComparator)
    comparator._event_normalizer = _EventNormalizer()

    caplog.set_level("WARNING")
    rows = comparator._process_drug_file(raw_file)

    assert len(rows) == 1
    assert rows[0]["drug_id"] == "drug-a"
    assert rows[0]["nct_id"] == "NCT00000001"
    assert "drug_id=drug-a" in caplog.text
    assert "studies[0]" in caplog.text
