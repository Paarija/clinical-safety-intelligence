"""
normalization/outcome_normalizer.py

Maps FAERS outcome codes (DE, HO, LT, DS, CA, RI, OT) to standardized
seriousness labels and flags.

Usage:
    from clinical_safety.normalization.outcome_normalizer import OutcomeNormalizer
    normalizer = OutcomeNormalizer()
    outc_df = normalizer.apply_to_dataframe(outc_df)
"""

from __future__ import annotations

import pandas as pd

from clinical_safety.common.config import get_config
from clinical_safety.common.logging import get_logger

logger = get_logger(__name__)

# Canonical mapping from FAERS outcome codes to labels and seriousness flags
OUTCOME_CODE_MAP: dict[str, dict] = {
    "DE": {"label": "Death", "is_serious": True, "sort_weight": 5},
    "HO": {"label": "Hospitalization", "is_serious": True, "sort_weight": 4},
    "LT": {"label": "Life-Threatening", "is_serious": True, "sort_weight": 3},
    "DS": {"label": "Disability", "is_serious": True, "sort_weight": 2},
    "CA": {"label": "Congenital Anomaly", "is_serious": True, "sort_weight": 2},
    "RI": {"label": "Required Intervention", "is_serious": True, "sort_weight": 1},
    "OT": {"label": "Other Serious", "is_serious": True, "sort_weight": 1},
}


class OutcomeNormalizer:
    """
    Adds standardized outcome labels and seriousness flags to the FAERS OUTC table.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._serious_codes = set(
            cfg.signal_thresholds.seriousness.serious_outcome_codes
        )

    def apply_to_dataframe(self, outc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add outcome_label, is_serious, and is_configured_serious columns.

        Args:
            outc_df: Parsed FAERS OUTC DataFrame with an outc_cod column.

        Returns:
            DataFrame with additional columns.
        """
        df = outc_df.copy()
        df["outc_cod_upper"] = df["outc_cod"].str.upper().str.strip()
        df["outcome_label"] = df["outc_cod_upper"].map(
            {k: v["label"] for k, v in OUTCOME_CODE_MAP.items()}
        ).fillna("Unknown")
        df["is_serious"] = df["outc_cod_upper"].isin(OUTCOME_CODE_MAP)
        df["is_configured_serious"] = df["outc_cod_upper"].isin(self._serious_codes)
        df = df.drop(columns=["outc_cod_upper"])
        logger.info(
            "OutcomeNormalizer: %d rows, %d serious (%.1f%%)",
            len(df),
            df["is_configured_serious"].sum(),
            100 * df["is_configured_serious"].mean(),
        )
        return df
