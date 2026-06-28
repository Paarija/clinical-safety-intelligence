"""
analytics/contingency_tables.py

Builds 2x2 contingency tables for each in-scope drug-event pair
from the deduplicated, normalized FAERS tables.

The 2x2 table for (drug D, event E):
    a = cases where D is suspect AND E is reported
    b = cases where D is suspect AND E is NOT reported
    c = cases where D is NOT suspect AND E is reported
    d = cases where D is NOT suspect AND E is NOT reported

Role code filter (from config):
    Default: PS (primary suspect) only.
    Sensitivity: PS + SS.

IMPORTANT:
    The contingency table is built from deduplicated cases only.
    One case = one unique drug + one unique event row after dedup.

Usage:
    from clinical_safety.analytics.contingency_tables import ContingencyTableBuilder
    builder = ContingencyTableBuilder()
    ct_df = builder.build(drug_df, reac_df)
"""

from __future__ import annotations

import pandas as pd

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import ContingencyTableError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)

_CONFIDENCE_ORDER = {
    "unmatched": 0,
    "fuzzy": 1,
    "alias": 2,
    "exact": 3,
}


def _confidence_rank(value: object) -> int:
    return _CONFIDENCE_ORDER.get(str(value).lower(), -1)


def _strongest_confidence(values: pd.Series) -> str:
    return max((str(v).lower() for v in values), key=_confidence_rank)


def _weakest_or_unmatched(values: list[str]) -> str:
    return min(values, key=_confidence_rank) if values else "unmatched"


def _meets_confidence(series: pd.Series, minimum: str) -> pd.Series:
    min_rank = _confidence_rank(minimum)
    return series.map(_confidence_rank) >= min_rank


def configured_signal_role_codes(sensitivity: bool = False) -> list[str]:
    """Return configured FAERS drug role codes for strict or sensitivity analysis."""
    sig_cfg = get_config().signal_thresholds.signal_detection
    configured = sig_cfg.sensitivity_role_codes if sensitivity else sig_cfg.role_code_filter
    return [role.upper() for role in configured]


def filter_signal_records(
    drug_df: pd.DataFrame,
    reac_df: pd.DataFrame,
    role_codes: list[str],
    min_mapping_confidence: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ["primaryid", "drug_id", "role_cod", "drug_mapping_confidence"]:
        if col not in drug_df.columns:
            raise ContingencyTableError(f"drug_df missing required column: {col}")
    for col in ["primaryid", "event_id", "event_mapping_confidence"]:
        if col not in reac_df.columns:
            raise ContingencyTableError(f"reac_df missing required column: {col}")

    role_set = [r.upper() for r in role_codes]
    drug_in_scope = drug_df[
        drug_df["drug_id"].notna()
        & drug_df["role_cod"].str.upper().isin(role_set)
        & _meets_confidence(drug_df["drug_mapping_confidence"], min_mapping_confidence)
    ][["primaryid", "drug_id", "drug_mapping_confidence"]].drop_duplicates()

    reac_in_scope = reac_df[
        reac_df["event_id"].notna()
        & _meets_confidence(reac_df["event_mapping_confidence"], min_mapping_confidence)
    ][["primaryid", "event_id", "event_mapping_confidence"]].drop_duplicates()

    return drug_in_scope, reac_in_scope


class ContingencyTableBuilder:
    """
    Builds drug-event contingency tables from normalized FAERS tables.

    Requires:
        drug_df: Normalized DRUG table with columns [primaryid, drug_id, role_cod, drug_mapping_confidence]
        reac_df: Normalized REAC table with columns [primaryid, event_id, event_mapping_confidence]
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        sig_cfg = cfg.signal_thresholds.signal_detection
        self._role_codes = configured_signal_role_codes(sensitivity=False)
        self._sensitivity_role_codes = configured_signal_role_codes(sensitivity=True)
        self._min_mapping_confidence = sig_cfg.min_mapping_confidence
        self._paths = paths or Paths()
        self._out_dir = self._paths.processed_signals
        self._out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ContingencyTableBuilder: role_codes=%s, sensitivity_role_codes=%s, min_confidence=%s",
            self._role_codes,
            self._sensitivity_role_codes,
            self._min_mapping_confidence,
        )

    def build(
        self,
        drug_df: pd.DataFrame,
        reac_df: pd.DataFrame,
        save: bool = True,
        sensitivity: bool = False,
    ) -> pd.DataFrame:
        """
        Build 2x2 contingency tables for all in-scope drug-event pairs.

        Args:
            drug_df    : Normalized FAERS DRUG table.
            reac_df    : Normalized FAERS REAC table.
            save       : If True, save result to processed/signals/contingency_tables.parquet.
            sensitivity: If True, use configured sensitivity role codes instead of the
                         default primary-suspect role-code filter.

        Returns:
            DataFrame with columns: drug_id, event_id, a, b, c, d, total_cases.
        """
        role_codes = self._sensitivity_role_codes if sensitivity else self._role_codes
        drug_in_scope, reac_in_scope = filter_signal_records(
            drug_df,
            reac_df,
            role_codes,
            self._min_mapping_confidence,
        )

        if drug_in_scope.empty:
            raise ContingencyTableError(
                "No in-scope drug records found after filtering. "
                "Check drug_scope.yaml aliases and role_code_filter config."
            )
        if reac_in_scope.empty:
            raise ContingencyTableError(
                "No in-scope event records found after filtering. "
                "Check event_scope.yaml preferred_terms."
            )

        # Universe of unique cases (primaryids)
        all_cases = set(drug_df["primaryid"].unique()) | set(reac_df["primaryid"].unique())
        total_cases = len(all_cases)
        logger.info(
            "Building contingency tables: %d total cases, %d in-scope drug records, "
            "%d in-scope event records, role_codes=%s",
            total_cases,
            len(drug_in_scope),
            len(reac_in_scope),
            role_codes,
        )

        # For each drug_id: set of cases matching the selected role-code filter
        drug_cases: dict[str, set] = (
            drug_in_scope.groupby("drug_id")["primaryid"]
            .apply(set)
            .to_dict()
        )

        # For each event_id: set of cases where it is reported
        event_cases: dict[str, set] = (
            reac_in_scope.groupby("event_id")["primaryid"]
            .apply(set)
            .to_dict()
        )

        drug_confidence_by_case = (
            drug_in_scope.groupby(["drug_id", "primaryid"])["drug_mapping_confidence"]
            .apply(_strongest_confidence)
            .to_dict()
        )
        event_confidence_by_case = (
            reac_in_scope.groupby(["event_id", "primaryid"])["event_mapping_confidence"]
            .apply(_strongest_confidence)
            .to_dict()
        )

        rows = []
        for drug_id, drug_set in drug_cases.items():
            for event_id, event_set in event_cases.items():
                a_cases = drug_set & event_set
                a = len(a_cases)
                b = len(drug_set - event_set)
                c = len(event_set - drug_set)
                d = total_cases - len(drug_set | event_set)
                rows.append({
                    "drug_id": drug_id,
                    "event_id": event_id,
                    "drug_mapping_confidence": _weakest_or_unmatched(
                        [drug_confidence_by_case[(drug_id, primaryid)] for primaryid in a_cases]
                    ),
                    "event_mapping_confidence": _weakest_or_unmatched(
                        [event_confidence_by_case[(event_id, primaryid)] for primaryid in a_cases]
                    ),
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                    "total_cases": total_cases,
                })

        ct_df = pd.DataFrame(rows)

        logger.info(
            "Contingency tables built: %d drug-event pairs", len(ct_df)
        )

        if save:
            filename = (
                "contingency_tables_sensitivity.parquet"
                if sensitivity
                else "contingency_tables.parquet"
            )
            out = self._out_dir / filename
            ct_df.to_parquet(out, index=False)
            logger.info("Saved contingency tables -> %s", out)

        return ct_df
