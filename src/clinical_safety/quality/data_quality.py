"""
quality/data_quality.py

Data quality report over parsed + normalized FAERS tables.

Covers:
  - Missingness rate per column per table
  - Schema violations (required columns absent)
  - Drug/event mapping confidence distribution
  - Deduplication summary (passed in from FAERSParser)

Output: JSON report saved to data/interim/quality_reports/data_quality_report.json

Usage:
    from clinical_safety.quality.data_quality import DataQualityReporter
    reporter = DataQualityReporter()
    report = reporter.run(tables, dedup_report, drug_normalizer, event_normalizer)
    reporter.save(report)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.parsing.faers_parser import REQUIRED_COLUMNS, DedupReport

logger = get_logger(__name__)


class DataQualityReporter:
    """Generates a data quality report for the FAERS ingestion pipeline."""

    def __init__(self, paths: Paths | None = None) -> None:
        self._out_dir = (paths or Paths()).interim_quality
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tables: dict[str, pd.DataFrame],
        dedup_report: DedupReport | None = None,
        drug_confidence_summary: dict[str, int] | None = None,
        event_confidence_summary: dict[str, int] | None = None,
        mapping_coverage_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Build the quality report dict.

        Args:
            tables                    : Output of FAERSParser.parse_all()
            dedup_report              : DedupReport from FAERSParser (optional)
            drug_confidence_summary   : DrugNormalizer.confidence_summary() (optional)
            event_confidence_summary  : EventNormalizer.confidence_summary() (optional)
            mapping_coverage_threshold: Optional minimum matched fraction for coverage checks.
        Returns:
            Report dict (also passed to save()).
        """
        report: dict[str, Any] = {
            "tables": {},
            "deduplication": dedup_report.as_dict() if dedup_report else None,
            "drug_mapping": drug_confidence_summary,
            "event_mapping": event_confidence_summary,
            "drug_mapping_coverage": (
                self._mapping_coverage(drug_confidence_summary, mapping_coverage_threshold)
                if drug_confidence_summary is not None
                else None
            ),
            "event_mapping_coverage": (
                self._mapping_coverage(event_confidence_summary, mapping_coverage_threshold)
                if event_confidence_summary is not None
                else None
            ),
        }

        for table_name, df in tables.items():
            report["tables"][table_name] = self._table_report(table_name, df)

        logger.info(
            "Data quality report built: %d tables, dedup=%s",
            len(tables),
            "yes" if dedup_report else "no",
        )
        return report

    def save(self, report: dict[str, Any]) -> Path:
        """Write the report as JSON."""
        out = self._out_dir / "data_quality_report.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Data quality report saved to %s", out)
        return out

    @staticmethod
    def _table_report(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
        """Missingness + schema check for one table."""
        required = REQUIRED_COLUMNS.get(table_name, [])
        missing_cols = [c for c in required if c not in df.columns]

        missingness = {
            col: round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        }
        # ponytail: only flag columns with any missingness to keep report readable
        missing_data = {k: v for k, v in missingness.items() if v > 0}

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "schema_violations": missing_cols,
            "missing_required_columns": missing_cols,
            "missingness_pct": missing_data,
            "high_missingness_cols": [k for k, v in missing_data.items() if v > 20],
        }

    @staticmethod
    def _mapping_coverage(
        confidence_summary: dict[str, int],
        threshold: float | None,
    ) -> dict[str, int | float | bool | str]:
        total = sum(confidence_summary.values())
        unmatched = confidence_summary.get("unmatched", 0)
        matched = total - unmatched
        coverage = matched / total if total else 0.0
        result: dict[str, int | float | bool | str] = {
            "scope": "all_unique_terms_seen",
            "total": total,
            "matched": matched,
            "unmatched": unmatched,
            "coverage_pct": round(coverage * 100, 2),
        }
        if threshold is not None:
            result["threshold_pct"] = round(threshold * 100, 2)
            result["passed"] = total > 0 and coverage >= threshold
        return result
