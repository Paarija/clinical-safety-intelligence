from __future__ import annotations

import pandas as pd

from clinical_safety.parsing.faers_parser import FAERSParser


def test_deduplicate_demo_tied_caseversion_keeps_latest_mfr_dt() -> None:
    demo = pd.DataFrame(
        [
            {"primaryid": "1001", "caseid": "case-a", "caseversion": "2", "mfr_dt": "20240101"},
            {"primaryid": "1002", "caseid": "case-a", "caseversion": "2", "mfr_dt": "20240201"},
            {"primaryid": "1003", "caseid": "case-a", "caseversion": "1", "mfr_dt": "20240301"},
        ]
    )

    deduped, report = FAERSParser._deduplicate_demo(demo)

    assert deduped.loc[0, "primaryid"] == "1002"
    assert report.unique_cases_before == 1
    assert report.unique_cases_after == 1
    assert report.duplicate_rows_removed == 2


def test_deduplicate_demo_tied_date_keeps_highest_primaryid() -> None:
    demo = pd.DataFrame(
        [
            {"primaryid": "2001", "caseid": "case-b", "caseversion": "3", "mfr_dt": "20240501"},
            {"primaryid": "2009", "caseid": "case-b", "caseversion": "3", "mfr_dt": "20240501"},
        ]
    )

    deduped, _ = FAERSParser._deduplicate_demo(demo)

    assert deduped.loc[0, "primaryid"] == "2009"
