"""
analytics/seriousness.py

Computes seriousness outcome summaries per drug-event pair from the
deduplicated, normalized FAERS OUTC table.

Outputs:
  - death_count
  - hospitalization_count
  - life_threatening_count
  - disability_count
  - any_serious_count
  - seriousness_rate  (any_serious_count / case_count)

IMPORTANT:
  Seriousness codes in FAERS are reporter-assigned.
  They represent reported outcomes, not drug-caused outcomes.
  seriousness_rate is NOT a drug risk estimate.

Usage:
    from clinical_safety.analytics.seriousness import SeriousnessSummarizer
    summarizer = SeriousnessSummarizer()
    seriousness_df = summarizer.summarize(drug_reac_df, outc_df)
"""

from __future__ import annotations

import pandas as pd

from clinical_safety.common.logging import get_logger

logger = get_logger(__name__)

# Outcome codes to individual columns
_OUTCOME_COLS = {
    "DE": "death_count",
    "HO": "hospitalization_count",
    "LT": "life_threatening_count",
    "DS": "disability_count",
    "RI": "required_intervention_count",
    "OT": "other_serious_count",
}


class SeriousnessSummarizer:
    """
    Aggregates FAERS outcome codes into per-drug-event seriousness summaries.

    Requires a merged DataFrame of (drug_id, event_id, primaryid) and the
    normalized outcome table with (primaryid, outc_cod).
    """

    def summarize(
        self,
        signal_cases_df: pd.DataFrame,
        outc_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute seriousness summary per drug-event pair.

        Args:
            signal_cases_df: DataFrame with [drug_id, event_id, primaryid]
                             (one row per case per drug-event pair).
            outc_df        : Normalized FAERS OUTC DataFrame with
                             [primaryid, outc_cod].

        Returns:
            DataFrame with [drug_id, event_id, case_count,
                            death_count, hospitalization_count,
                            life_threatening_count, disability_count,
                            any_serious_count, seriousness_rate].
        """
        # Pivot outcome codes: one column per outcome type, 1/0 per case
        outc_pivot = outc_df[["primaryid", "outc_cod"]].copy()
        outc_pivot["outc_cod"] = outc_pivot["outc_cod"].str.upper().str.strip()
        for code, col in _OUTCOME_COLS.items():
            outc_pivot[col] = (outc_pivot["outc_cod"] == code).astype(int)

        # Aggregate: max per primaryid (avoid double-counting multi-outcome cases)
        outc_per_case = (
            outc_pivot
            .groupby("primaryid")[list(_OUTCOME_COLS.values())]
            .max()
            .reset_index()
        )

        # Merge with signal cases
        merged = signal_cases_df[["drug_id", "event_id", "primaryid"]].merge(
            outc_per_case, on="primaryid", how="left"
        ).fillna(0)

        # Simpler approach: compute any_serious in merged frame first
        merged["any_serious"] = merged[list(_OUTCOME_COLS.values())].max(axis=1)

        agg = (
            merged
            .groupby(["drug_id", "event_id"])
            .agg(
                case_count=("primaryid", "nunique"),
                death_count=("death_count", "sum"),
                hospitalization_count=("hospitalization_count", "sum"),
                life_threatening_count=("life_threatening_count", "sum"),
                disability_count=("disability_count", "sum"),
                any_serious_count=("any_serious", "sum"),
            )
            .reset_index()
        )

        # Seriousness rate
        agg["seriousness_rate"] = (
            agg["any_serious_count"] / agg["case_count"].replace(0, float("nan"))
        ).round(4)

        logger.info(
            "Seriousness summary: %d drug-event pairs, avg seriousness_rate=%.2f%%",
            len(agg),
            100 * agg["seriousness_rate"].mean() if not agg.empty else 0,
        )
        return agg
