"""
analytics/disproportionality.py

Computes ROR, PRR, confidence intervals, and p-values for drug-event pairs
detected in the FAERS data.

Formulas:
  Given a 2x2 contingency table:
    a = reports with DRUG and EVENT
    b = reports with DRUG and NOT EVENT
    c = reports with NOT DRUG and EVENT
    d = reports with NOT DRUG and NOT EVENT

  ROR = (a * d) / (b * c)
  PRR = (a / (a + b)) / (c / (c + d))
  ROR 95% CI: exp(log(ROR) ± 1.96 * sqrt(1/a + 1/b + 1/c + 1/d))

IMPORTANT:
  All metrics are measures of *disproportionate reporting*, not incidence.
  These values cannot establish causality.

Usage:
    from clinical_safety.analytics.disproportionality import DisproportionalityCalculator
    calc = DisproportionalityCalculator()
    results_df = calc.compute(contingency_df)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from clinical_safety.common.config import get_config
from clinical_safety.common.logging import get_logger

logger = get_logger(__name__)

# Small epsilon to avoid division by zero / log(0)
_EPS = 0.5


class DisproportionalityCalculator:
    """
    Computes ROR, PRR, CIs, and chi-squared p-values from a contingency table.

    Input DataFrame columns expected:
        drug_id   : str
        event_id  : str
        a         : int  (drug AND event)
        b         : int  (drug AND NOT event)
        c         : int  (NOT drug AND event)
        d         : int  (NOT drug AND NOT event)
    """

    def __init__(self) -> None:
        cfg = get_config()
        disp_cfg = cfg.signal_thresholds.disproportionality
        sig_cfg = cfg.signal_thresholds.signal_detection

        self._ci_z = self._z_for_coverage(cfg.signal_thresholds.signal_detection.ci_coverage)
        self._prr_enabled = disp_cfg.prr_enabled
        self._chi2_p_threshold = disp_cfg.chi2_p_threshold
        self._min_case_count = sig_cfg.min_case_count
        self._ror_lower_ci_threshold = sig_cfg.ror_lower_ci_threshold

        logger.info(
            "DisproportionalityCalculator: CI z=%.3f, min_cases=%d, ROR_CI_threshold=%.1f",
            self._ci_z,
            self._min_case_count,
            self._ror_lower_ci_threshold,
        )

    @staticmethod
    def _z_for_coverage(coverage: float) -> float:
        """Convert CI coverage fraction to z-score (e.g. 0.95 -> 1.96)."""
        alpha = 1 - coverage
        return float(stats.norm.ppf(1 - alpha / 2))

    def compute(self, contingency_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all disproportionality metrics for each drug-event pair.

        Args:
            contingency_df: DataFrame with columns drug_id, event_id, a, b, c, d.

        Returns:
            DataFrame with columns:
                drug_id, event_id, case_count,
                ror, ror_lower_ci, ror_upper_ci,
                prr (if enabled), chi2_p_value,
                signal_flagged (bool)
        """
        df = contingency_df.copy()

        # Ensure integer columns
        for col in ["a", "b", "c", "d"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # case_count = a (reports with this drug AND this event)
        df["case_count"] = df["a"]

        # Apply Haldane-Anscombe correction: add 0.5 to all cells when any cell is 0
        # (standard practice for zero-cell contingency tables)
        for col in ["a", "b", "c", "d"]:
            df[f"{col}_adj"] = df[col].where(
                (df["a"] > 0) & (df["b"] > 0) & (df["c"] > 0) & (df["d"] > 0),
                df[col] + _EPS,
            )

        # ROR = (a * d) / (b * c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["ror"] = (df["a_adj"] * df["d_adj"]) / (df["b_adj"] * df["c_adj"])

        # ROR log-normal 95% CI
        # Var(log ROR) = 1/a + 1/b + 1/c + 1/d
        df["log_ror"] = np.log(df["ror"].replace(0, np.nan))
        df["se_log_ror"] = np.sqrt(
            1 / df["a_adj"] + 1 / df["b_adj"] + 1 / df["c_adj"] + 1 / df["d_adj"]
        )
        df["ror_lower_ci"] = np.exp(df["log_ror"] - self._ci_z * df["se_log_ror"])
        df["ror_upper_ci"] = np.exp(df["log_ror"] + self._ci_z * df["se_log_ror"])

        # PRR = [a/(a+b)] / [c/(c+d)]
        if self._prr_enabled:
            df["prr"] = (
                (df["a_adj"] / (df["a_adj"] + df["b_adj"])) /
                (df["c_adj"] / (df["c_adj"] + df["d_adj"]))
            )
        else:
            df["prr"] = np.nan

        # Chi-squared p-value (2x2 contingency table)
        df["chi2_p_value"] = df.apply(self._chi2_pvalue, axis=1)

        # Signal flagging
        df["signal_flagged"] = (
            (df["case_count"] >= self._min_case_count) &
            (df["ror_lower_ci"] >= self._ror_lower_ci_threshold)
        )

        # Clean up intermediate columns
        adj_cols = [c for c in df.columns if c.endswith("_adj") or c in ("log_ror", "se_log_ror")]
        df = df.drop(columns=adj_cols)

        # Round to readable precision
        for col in ["ror", "ror_lower_ci", "ror_upper_ci", "prr"]:
            if col in df.columns:
                df[col] = df[col].round(3)
        df["chi2_p_value"] = df["chi2_p_value"].round(4)

        flagged = df["signal_flagged"].sum()
        logger.info(
            "Disproportionality: %d drug-event pairs computed, %d flagged as signals",
            len(df),
            flagged,
        )
        return df

    @staticmethod
    def _chi2_pvalue(row: pd.Series) -> float:
        """Compute 2-tailed chi-squared p-value for one contingency row."""
        try:
            table = [[row["a"], row["b"]], [row["c"], row["d"]]]
            chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
            return float(p)
        except Exception:
            return float("nan")
