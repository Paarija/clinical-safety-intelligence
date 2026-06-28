"""
normalization/event_normalizer.py

Maps raw FAERS reaction preferred terms (pt field) to normalized event family
IDs defined in configs/event_scope.yaml.

Matching pipeline:
  1. Exact match against preferred_terms list
  2. Fuzzy match against all preferred_terms

Also provides outcome-code-based mapping for the "hospitalization" family
which is derived from OUTC (HO code) rather than REAC.

Usage:
    from clinical_safety.normalization.event_normalizer import EventNormalizer
    normalizer = EventNormalizer()
    reac_df["event_id"], reac_df["event_mapping_confidence"] = normalizer.normalize_series(reac_df["pt"])
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import EventNormalizationError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import MappingConfidence

logger = get_logger(__name__)


def _clean(term: str) -> str:
    return term.lower().strip() if isinstance(term, str) else ""


class EventNormalizer:
    """
    Normalizes raw FAERS reaction terms and trial AE terms to event family IDs.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._event_cfg = cfg.event_scope
        self._match_cfg = cfg.event_scope.matching
        self._paths = paths or Paths()
        self._out_dir = self._paths.interim_normalized
        self._out_dir.mkdir(parents=True, exist_ok=True)

        if not self._event_cfg.event_families:
            raise EventNormalizationError(
                "Event scope config is empty. Add event families to configs/event_scope.yaml."
            )

        # Build lookup structures
        self._exact_map: dict[str, str] = {}        # clean_term -> event_id
        self._all_terms: list[tuple[str, str]] = []  # (clean_term, event_id)
        self._outcome_code_map: dict[str, str] = {}  # outc_code -> event_id

        for fam in self._event_cfg.event_families:
            for term in fam.preferred_terms:
                ct = _clean(term)
                self._exact_map[ct] = fam.id
                self._all_terms.append((ct, fam.id))
            for code in fam.outcome_codes:
                self._outcome_code_map[code.upper()] = fam.id

        self._fuzzy_threshold = self._match_cfg.fuzzy_threshold / 100.0
        self._audit: dict[str, tuple[str | None, MappingConfidence]] = {}

        logger.info(
            "EventNormalizer initialized: %d families, %d preferred terms, fuzzy=%.0f%%",
            len(self._event_cfg.event_families),
            len(self._all_terms),
            self._match_cfg.fuzzy_threshold,
        )

    def normalize(self, raw_term: str) -> tuple[str | None, MappingConfidence]:
        """Normalize a single raw FAERS reaction term."""
        if raw_term in self._audit:
            return self._audit[raw_term]

        result = self._match(raw_term)
        self._audit[raw_term] = result
        return result

    def normalize_series(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Normalize a whole pandas Series of reaction terms."""
        results = series.apply(self.normalize)
        event_ids = results.apply(lambda x: x[0])
        confidences = results.apply(lambda x: x[1].value)
        return event_ids, confidences

    def normalize_outcome_codes(self, series: pd.Series) -> pd.Series:
        """
        Map FAERS outcome codes (DE, HO, etc.) to event_ids.
        Returns a Series of event_ids (or None where no mapping exists).
        """
        return series.str.upper().map(self._outcome_code_map)

    def apply_to_dataframe(self, df: pd.DataFrame, col: str = "pt") -> pd.DataFrame:
        """Add event_id and event_mapping_confidence columns to a copy of df."""
        df = df.copy()
        event_ids, confidences = self.normalize_series(df[col])
        df["event_id"] = event_ids
        df["event_mapping_confidence"] = confidences
        n_in_scope = df["event_id"].notna().sum()
        logger.info(
            "EventNormalizer: %d/%d reaction rows mapped to in-scope event (%.1f%%)",
            n_in_scope, len(df), 100 * n_in_scope / max(len(df), 1),
        )
        return df

    def _match(self, raw_term: str) -> tuple[str | None, MappingConfidence]:
        if not raw_term or not isinstance(raw_term, str):
            return None, MappingConfidence.UNMATCHED

        cleaned = _clean(raw_term)

        # 1. Exact
        if cleaned in self._exact_map:
            return self._exact_map[cleaned], MappingConfidence.EXACT

        # 2. Fuzzy
        best_score = 0.0
        best_id: str | None = None
        for term, event_id in self._all_terms:
            score = SequenceMatcher(None, cleaned, term).ratio()
            if score > best_score:
                best_score = score
                best_id = event_id
        if best_score >= self._fuzzy_threshold and best_id:
            return best_id, MappingConfidence.FUZZY

        return None, MappingConfidence.UNMATCHED

    def save_audit(self) -> Path:
        """Save event mapping audit to interim/normalized/event_map_audit.parquet."""
        rows = [
            {"raw_term": raw, "event_id": eid, "mapping_confidence": conf.value}
            for raw, (eid, conf) in self._audit.items()
        ]
        df = pd.DataFrame(rows)
        out = self._out_dir / "event_map_audit.parquet"
        df.to_parquet(out, index=False)
        n_matched = df["event_id"].notna().sum()
        logger.info(
            "Event audit saved: %d unique terms, %d matched (%.1f%%)",
            len(df), n_matched, 100 * n_matched / max(len(df), 1),
        )
        return out

    def confidence_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {c.value: 0 for c in MappingConfidence}
        for _, conf in self._audit.values():
            counts[conf.value] = counts.get(conf.value, 0) + 1
        return counts

    def validate_mapping_coverage(
        self, threshold: float = 0.90
    ) -> dict[str, int | float | bool]:
        """Return coverage stats for unique audit entries."""
        total = len(self._audit)
        matched = sum(1 for event_id, _ in self._audit.values() if event_id is not None)
        coverage = matched / total if total else 0.0
        return {
            "total": total,
            "matched": matched,
            "coverage_pct": round(coverage * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
            "passed": total > 0 and coverage >= threshold,
        }
