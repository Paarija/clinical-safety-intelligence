"""
normalization/drug_normalizer.py

Maps raw FAERS drug name strings to normalized drug IDs defined in
configs/drug_scope.yaml.

Matching pipeline (in order):
  1. Exact match (after lowercasing and stripping)
  2. Alias match (any configured alias)
  3. Strip dosage suffix and retry exact + alias
  4. Fuzzy match using SequenceMatcher (if above fuzzy_threshold)
  5. Unmatched

Emits a mapping audit table with one row per unique raw drug name seen.

Usage:
    from clinical_safety.normalization.drug_normalizer import DrugNormalizer
    normalizer = DrugNormalizer()
    drug_df["drug_id"], drug_df["mapping_confidence"] = normalizer.normalize_series(drug_df["drugname"])
    normalizer.save_audit()
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import DrugNormalizationError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import MappingConfidence

logger = get_logger(__name__)

# Regex to strip common dosage/formulation suffixes
_DOSAGE_RE = re.compile(
    r"\s+\d+[\.,]?\d*\s*(?:mg|mcg|ug|ml|g|iu|units?|%|mg/ml|mg/dl|nmol|mmol).*$",
    flags=re.IGNORECASE,
)
_ROUTE_SUFFIXES = re.compile(
    r"\s+(?:injection|oral|subcutaneous|intravenous|extended[\-\s]release|er|xr|sr|la|hcl|hydrochloride)$",
    flags=re.IGNORECASE,
)


def _clean(name: str) -> str:
    """Lowercase, strip whitespace."""
    return name.lower().strip() if isinstance(name, str) else ""


def _strip_dosage(name: str) -> str:
    """Remove dosage suffix and route suffix."""
    s = _DOSAGE_RE.sub("", name)
    s = _ROUTE_SUFFIXES.sub("", s)
    return s.strip()


class DrugNormalizer:
    """
    Normalizes raw FAERS drug name strings to internal drug IDs.

    Thread-safety: not thread-safe (uses a mutable audit dict).
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._drug_cfg = cfg.drug_scope
        self._match_cfg = cfg.drug_scope.matching
        self._paths = paths or Paths()
        self._out_dir = self._paths.interim_normalized
        self._out_dir.mkdir(parents=True, exist_ok=True)

        if not self._drug_cfg.drugs:
            raise DrugNormalizationError(
                "Drug scope config is empty. Add at least one drug to configs/drug_scope.yaml."
            )

        # Build lookup structures
        self._exact_map: dict[str, str] = {}    # clean_name -> drug_id
        self._alias_map: dict[str, str] = {}    # clean_alias -> drug_id
        self._drug_ids: list[str] = []
        self._all_aliases: list[tuple[str, str]] = []  # (clean_alias, drug_id)

        for entry in self._drug_cfg.drugs:
            self._drug_ids.append(entry.id)
            self._exact_map[_clean(entry.normalized_name)] = entry.id
            for alias in entry.aliases:
                ca = _clean(alias)
                self._alias_map[ca] = entry.id
                self._all_aliases.append((ca, entry.id))

        self._fuzzy_threshold = self._match_cfg.fuzzy_threshold / 100.0
        # Audit: raw_name -> (drug_id, confidence)
        self._audit: dict[str, tuple[str | None, MappingConfidence]] = {}

        logger.info(
            "DrugNormalizer initialized: %d drugs, %d aliases, fuzzy threshold=%.0f%%",
            len(self._drug_ids),
            len(self._alias_map),
            self._match_cfg.fuzzy_threshold,
        )

    def normalize(self, raw_name: str) -> tuple[str | None, MappingConfidence]:
        """
        Normalize a single raw drug name.

        Returns:
            (drug_id, confidence) — drug_id is None if unmatched.
        """
        if raw_name in self._audit:
            return self._audit[raw_name]

        result = self._match(raw_name)
        self._audit[raw_name] = result
        return result

    def normalize_series(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Normalize a whole pandas Series of raw drug names.

        Returns:
            (drug_id_series, confidence_series)
        """
        results = series.apply(self.normalize)
        drug_ids = results.apply(lambda x: x[0])
        confidences = results.apply(lambda x: x[1].value)
        return drug_ids, confidences

    def _match(self, raw_name: str) -> tuple[str | None, MappingConfidence]:
        """Run matching pipeline."""
        if not raw_name or not isinstance(raw_name, str):
            return None, MappingConfidence.UNMATCHED

        cleaned = _clean(raw_name)

        # 1. Exact match on normalized name
        if cleaned in self._exact_map:
            return self._exact_map[cleaned], MappingConfidence.EXACT

        # 2. Alias match
        if cleaned in self._alias_map:
            return self._alias_map[cleaned], MappingConfidence.ALIAS

        # 3. Strip dosage suffix and retry
        stripped = _strip_dosage(cleaned)
        if stripped and stripped != cleaned:
            if stripped in self._exact_map:
                return self._exact_map[stripped], MappingConfidence.ALIAS
            if stripped in self._alias_map:
                return self._alias_map[stripped], MappingConfidence.ALIAS

        # 4. Fuzzy match
        best_score = 0.0
        best_id: str | None = None
        for alias, drug_id in self._all_aliases:
            score = SequenceMatcher(None, cleaned, alias).ratio()
            if score > best_score:
                best_score = score
                best_id = drug_id
        if best_score >= self._fuzzy_threshold and best_id:
            return best_id, MappingConfidence.FUZZY

        return None, MappingConfidence.UNMATCHED

    def apply_to_dataframe(self, df: pd.DataFrame, col: str = "drugname") -> pd.DataFrame:
        """
        Add drug_id and drug_mapping_confidence columns to a DataFrame in-place copy.

        Args:
            df : DataFrame containing a column with raw drug names.
            col: Name of the column with raw drug names (default: "drugname").

        Returns:
            DataFrame with two new columns: drug_id, drug_mapping_confidence.
        """
        df = df.copy()
        drug_ids, confidences = self.normalize_series(df[col])
        df["drug_id"] = drug_ids
        df["drug_mapping_confidence"] = confidences
        n_in_scope = (df["drug_mapping_confidence"] != MappingConfidence.UNMATCHED.value).sum()
        logger.info(
            "DrugNormalizer: %d/%d rows matched to in-scope drug (%.1f%%)",
            n_in_scope, len(df), 100 * n_in_scope / max(len(df), 1),
        )
        return df

    def save_audit(self) -> Path:
        """Save mapping audit table to interim/normalized/drug_map_audit.parquet."""
        rows = [
            {"raw_name": raw, "drug_id": drug_id, "mapping_confidence": conf.value}
            for raw, (drug_id, conf) in self._audit.items()
        ]
        df = pd.DataFrame(rows)
        out = self._out_dir / "drug_map_audit.parquet"
        df.to_parquet(out, index=False)
        n_matched = df["drug_id"].notna().sum()
        logger.info(
            "Drug audit saved: %d unique raw names, %d matched (%.1f%%), %d unmatched",
            len(df), n_matched,
            100 * n_matched / max(len(df), 1),
            len(df) - n_matched,
        )
        return out

    def confidence_summary(self) -> dict[str, int]:
        """Return count of each confidence level in the audit."""
        counts: dict[str, int] = {c.value: 0 for c in MappingConfidence}
        for _, conf in self._audit.values():
            counts[conf.value] = counts.get(conf.value, 0) + 1
        return counts

    def validate_mapping_coverage(
        self, threshold: float = 0.90
    ) -> dict[str, int | float | bool]:
        """Return coverage stats for unique audit entries."""
        total = len(self._audit)
        matched = sum(1 for drug_id, _ in self._audit.values() if drug_id is not None)
        coverage = matched / total if total else 0.0
        return {
            "total": total,
            "matched": matched,
            "coverage_pct": round(coverage * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
            "passed": total > 0 and coverage >= threshold,
        }
