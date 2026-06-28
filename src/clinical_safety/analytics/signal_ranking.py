"""
analytics/signal_ranking.py

Joins disproportionality metrics, seriousness summaries, and (where available)
trial event-rate data into a ranked candidate signal shortlist.

Ranking score (configurable):
    rank_score = ror_lower_ci * log1p(case_count) * (1 + seriousness_rate)

Signals are then filtered by configured thresholds and saved as a
ranked candidate shortlist.

Usage:
    from clinical_safety.analytics.signal_ranking import SignalRanker
    ranker = SignalRanker()
    shortlist = ranker.rank(disp_df, seriousness_df)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from clinical_safety.common.exceptions import ClinicalSafetyError
from clinical_safety.common.config import get_config
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)

class SignalRanker:
    """
    Produces a ranked candidate signal shortlist from disproportionality
    and seriousness DataFrames.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._sig_cfg = cfg.signal_thresholds.signal_detection
        self._paths = paths or Paths()
        self._out_dir = self._paths.processed_signals
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def rank(
        self,
        disp_df: pd.DataFrame,
        seriousness_df: pd.DataFrame,
        trial_df: pd.DataFrame | None = None,
        evidence_window: str = "unknown",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Join metrics and rank drug-event pairs.

        Args:
            disp_df       : Output of DisproportionalityCalculator.compute()
            seriousness_df: Output of SeriousnessSummarizer.summarize()
            trial_df      : Optional trial event-rate comparison table.
            evidence_window: Label for the FAERS quarter/period.
            save          : Save shortlist to processed/signals/.

        Returns:
            Ranked DataFrame with all signal columns + rank_score.
        """
        # Merge disproportionality + seriousness
        merged = disp_df.merge(
            seriousness_df,
            on=["drug_id", "event_id"],
            how="left",
            suffixes=("", "_seriousness"),
        )

        # Use case_count from disproportionality (a-column) as authoritative
        if "case_count_seriousness" in merged.columns:
            merged = merged.drop(columns=["case_count_seriousness"])

        # Merge trial data if available
        if trial_df is not None and not trial_df.empty:
            trial_summary = trial_df.groupby(["drug_id", "event_id"]).agg(
                trial_evidence_available=("nct_id", lambda x: x.notna().any()),
                n_matching_trials=("nct_id", "nunique"),
            ).reset_index()
            merged = merged.merge(trial_summary, on=["drug_id", "event_id"], how="left")
            merged["trial_evidence_available"] = merged["trial_evidence_available"].fillna(False)
            merged["n_matching_trials"] = merged["n_matching_trials"].fillna(0).astype(int)
        else:
            merged["trial_evidence_available"] = False
            merged["n_matching_trials"] = 0

        # Add evidence window label
        merged["evidence_window"] = evidence_window

        # Apply threshold filters
        min_cases = self._sig_cfg.min_case_count
        ror_ci_thr = self._ror_lower_ci_threshold = self._sig_cfg.ror_lower_ci_threshold

        filtered = merged[
            (merged["case_count"] >= min_cases) &
            (merged["ror_lower_ci"] >= ror_ci_thr)
        ].copy()

        if filtered.empty:
            logger.warning(
                "No signals passed thresholds (min_cases=%d, ror_ci>=%s). "
                "Consider lowering thresholds in signal_thresholds.yaml.",
                min_cases, ror_ci_thr,
            )

        # Compute ranking score
        filtered["rank_score"] = (
            filtered["ror_lower_ci"].fillna(0) *
            np.log1p(filtered["case_count"].fillna(0)) *
            (1 + filtered["seriousness_rate"].fillna(0))
        ).round(4)

        # Sort by rank_score descending
        filtered = filtered.sort_values("rank_score", ascending=False).reset_index(drop=True)
        filtered.index.name = "rank"
        filtered.index = filtered.index + 1  # 1-indexed rank

        logger.info(
            "Signal ranking complete: %d/%d pairs passed thresholds. Top signal: %s + %s (ROR_CI=%.2f, n=%d)",
            len(filtered),
            len(merged),
            filtered.iloc[0]["drug_id"] if not filtered.empty else "—",
            filtered.iloc[0]["event_id"] if not filtered.empty else "—",
            filtered.iloc[0]["ror_lower_ci"] if not filtered.empty else 0,
            filtered.iloc[0]["case_count"] if not filtered.empty else 0,
        )

        if save:
            out = self._out_dir / "candidate_signals.parquet"
            filtered.reset_index().to_parquet(out, index=False)
            logger.info("Candidate signal shortlist saved -> %s", out)

            # Also save a human-readable CSV
            csv_out = self._out_dir / "candidate_signals.csv"
            filtered.reset_index().to_csv(csv_out, index=False)

        return filtered

def main() -> int:
    """Run the signal-ranking pipeline CLI."""
    try:
        from clinical_safety.common.logging import configure_logging

        configure_logging()
        from clinical_safety.analytics.contingency_tables import (
            ContingencyTableBuilder,
            configured_signal_role_codes,
            filter_signal_records,
        )
        from clinical_safety.analytics.disproportionality import DisproportionalityCalculator
        from clinical_safety.analytics.seriousness import SeriousnessSummarizer
        # 1. Paths and Config
        paths = Paths()

        # 2. Check if normalized data is available
        drug_file = paths.interim_normalized / "faers_drug_normalized.parquet"
        reac_file = paths.interim_normalized / "faers_reac_normalized.parquet"
        outc_file = paths.interim_normalized / "faers_outc_normalized.parquet"

        if not drug_file.exists() or not reac_file.exists() or not outc_file.exists():
            print(
                "\nError: Normalized FAERS tables not found under data/interim/normalized/.",
                file=sys.stderr,
            )
            print(
                "Please run the FAERS ingestion pipeline first: "
                "python -m clinical_safety.acquisition.faers_source",
                file=sys.stderr,
            )
            return 1

        drug_df = pd.read_parquet(drug_file)
        reac_df = pd.read_parquet(reac_file)
        outc_df = pd.read_parquet(outc_file)

        # 3. Contingency tables
        ct_builder = ContingencyTableBuilder(paths=paths)
        ct_df = ct_builder.build(drug_df, reac_df)

        # 4. Disproportionality
        calc = DisproportionalityCalculator()
        disp_df = calc.compute(ct_df)

        # 5. Seriousness
        sig_cfg = get_config().signal_thresholds.signal_detection
        drug_in_scope, reac_in_scope = filter_signal_records(
            drug_df,
            reac_df,
            configured_signal_role_codes(sensitivity=False),
            sig_cfg.min_mapping_confidence,
        )
        signal_cases_df = drug_in_scope[["primaryid", "drug_id"]].merge(
            reac_in_scope[["primaryid", "event_id"]],
            on="primaryid",
        )
        summarizer = SeriousnessSummarizer()
        seriousness_df = summarizer.summarize(signal_cases_df, outc_df)

        # 6. Trial data comparison if available
        trial_file = paths.processed_analytics / "trial_comparison.parquet"
        trial_df = None
        if trial_file.exists():
            trial_df = pd.read_parquet(trial_file)
            print("Loaded trial comparison data.")
        else:
            print("Trial comparison data not found (will rank without trial evidence).")

        # 7. Rank signals
        ranker = SignalRanker(paths=paths)
        cfg = get_config()
        quarter = cfg.data_sources.faers.quarter

        shortlist = ranker.rank(
            disp_df=disp_df,
            seriousness_df=seriousness_df,
            trial_df=trial_df,
            evidence_window=quarter,
        )

        print(
            f"\nSignal Ranking Pipeline completed successfully! "
            f"Shortlisted {len(shortlist)} signals."
        )
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except (ClinicalSafetyError, ValueError, OSError, RuntimeError) as exc:
        print(f"\nSignal Ranking Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
