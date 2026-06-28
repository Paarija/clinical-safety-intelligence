"""
parsing/faers_parser.py

Parses the four FAERS ASCII flat files (DEMO, DRUG, REAC, OUTC) into
clean Pandas DataFrames, then deduplicates FAERS cases by keeping the
latest version per caseid.

Responsibilities:
  - Read $-delimited files with latin-1 encoding
  - Standardize column names to snake_case
  - Validate required columns per file type
  - Deduplicate: keep max(caseversion) per caseid
  - Save each table as Parquet to data/interim/parsed/
  - Return deduplication report

Usage:
    from clinical_safety.parsing.faers_parser import FAERSParser
    parser = FAERSParser()
    tables = parser.parse_all(file_map)   # file_map from FAERSSource.acquire()
    report = parser.dedup_report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import ParseError, SchemaValidationError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)

# ── Required columns per FAERS file type ─────────────────────────────────────

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "DEMO": ["primaryid", "caseid", "caseversion"],
    "DRUG": ["primaryid", "caseid", "drugname", "role_cod"],
    "REAC": ["primaryid", "caseid", "pt"],
    "OUTC": ["primaryid", "caseid", "outc_cod"],
    "INDI": ["primaryid", "caseid", "indi_pt"],
}

# Columns to keep per table (superset; extras are dropped silently)
KEEP_COLUMNS: dict[str, list[str]] = {
    "DEMO": [
        "primaryid", "caseid", "caseversion", "fda_dt", "rept_dt", "mfr_dt",
        "age", "age_cod", "sex", "wt", "wt_cod", "country", "occp_cod",
        "reporter_country", "rept_cod", "mfr_sndr",
    ],
    "DRUG": [
        "primaryid", "caseid", "drug_seq", "role_cod", "drugname",
        "prod_ai", "val_vbm", "route", "dose_vbm", "cum_dose_unit",
        "dose_amt", "dose_unit", "dose_freq", "lot_num", "nda_num",
        "exp_dt", "dechal", "rechal",
    ],
    "REAC": ["primaryid", "caseid", "pt", "drug_rec_act"],
    "OUTC": ["primaryid", "caseid", "outc_cod"],
    "INDI": ["primaryid", "caseid", "indi_drug_seq", "indi_pt"],
}


@dataclass
class DedupReport:
    """Summary of FAERS case deduplication."""
    table: str = "DEMO"
    total_rows_before: int = 0
    unique_cases_before: int = 0
    unique_cases_after: int = 0
    duplicate_rows_removed: int = 0
    duplicate_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "total_rows_before": self.total_rows_before,
            "unique_cases_before": self.unique_cases_before,
            "unique_cases_after": self.unique_cases_after,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "duplicate_rate_pct": round(self.duplicate_rate * 100, 2),
        }


class FAERSParser:
    """
    Parses FAERS ASCII flat files into clean, deduplicated Parquet tables.

    Deduplication strategy:
      - DEMO: Group by caseid, keep highest caseversion; ties keep latest mfr_dt,
        then highest primaryid. This deterministic order avoids run-dependent case
        selection when FAERS ships duplicate versions.
      - DRUG/REAC/OUTC/INDI: Filtered to deduplicated primaryid set.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._faers_cfg = cfg.data_sources.faers
        self._paths = paths or Paths()
        self._out_dir = self._paths.interim_parsed
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_report: DedupReport | None = None
        self._valid_primaryids: set[str] | None = None

    def parse_all(self, file_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
        """
        Parse all available FAERS files and save as Parquet.

        Args:
            file_map: Dict from FAERSSource.acquire() mapping prefix → Path.

        Returns:
            Dict mapping prefix → deduplicated DataFrame.
        """
        tables: dict[str, pd.DataFrame] = {}

        # DEMO must be parsed first (establishes valid primaryid set)
        if "DEMO" not in file_map:
            raise ParseError("DEMO file is required for FAERS deduplication but was not found.")

        demo_df = self._parse_file("DEMO", file_map["DEMO"])
        demo_deduped, self.dedup_report = self._deduplicate_demo(demo_df)
        self._valid_primaryids = set(demo_deduped["primaryid"].astype(str))
        tables["DEMO"] = demo_deduped
        self._save_parquet(demo_deduped, "faers_demo")

        logger.info(
            "DEMO dedup: %d raw rows -> %d unique cases (removed %d duplicates, %.1f%%)",
            self.dedup_report.total_rows_before,
            self.dedup_report.unique_cases_after,
            self.dedup_report.duplicate_rows_removed,
            self.dedup_report.duplicate_rate * 100,
        )

        # Parse remaining tables filtered to valid primaryids
        for prefix in ["DRUG", "REAC", "OUTC", "INDI"]:
            if prefix not in file_map:
                logger.debug("File %s not in file_map — skipping.", prefix)
                continue
            df = self._parse_file(prefix, file_map[prefix])
            df_filtered = self._filter_to_valid_cases(df, prefix)
            tables[prefix] = df_filtered
            self._save_parquet(df_filtered, f"faers_{prefix.lower()}")

        return tables

    def _parse_file(self, file_type: str, path: Path) -> pd.DataFrame:
        """Read a single FAERS ASCII file into a DataFrame."""
        logger.info("Parsing %s from %s ...", file_type, path.name)
        try:
            df = pd.read_csv(
                path,
                delimiter=self._faers_cfg.delimiter,
                encoding=self._faers_cfg.encoding,
                low_memory=False,
                dtype=str,           # read everything as str; cast later
            )
        except Exception as exc:
            raise ParseError(f"Failed to read FAERS {file_type} file '{path}': {exc}") from exc

        # Normalize column names: strip, lower, replace spaces/dots with _
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\.]+", "_", regex=True)
        )

        # Validate required columns
        required = REQUIRED_COLUMNS.get(file_type, [])
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise SchemaValidationError(
                table_name=f"FAERS_{file_type}",
                details=f"Missing required columns: {missing_cols}. Found: {list(df.columns)[:20]}",
            )

        # Keep only known columns (ignore extra cols gracefully)
        keep = [c for c in KEEP_COLUMNS.get(file_type, required) if c in df.columns]
        df = df[keep].copy()

        # Strip whitespace from all string values
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].str.strip()

        logger.info("  %s: %d rows, %d columns", file_type, len(df), len(df.columns))
        return df

    @staticmethod
    def _deduplicate_demo(demo_df: pd.DataFrame) -> tuple[pd.DataFrame, DedupReport]:
        """
        Keep only the latest version of each case.

        Sort by caseversion, manufacturer date, then primaryid so tied versions
        choose the same row across runs and input orderings.
        """
        report = DedupReport(table="DEMO")
        report.total_rows_before = len(demo_df)
        report.unique_cases_before = demo_df["caseid"].nunique()

        # Ensure tie-break fields are numeric for deterministic comparison.
        demo_df = demo_df.copy()
        demo_df["caseversion_num"] = pd.to_numeric(demo_df["caseversion"], errors="coerce").fillna(0)
        if "mfr_dt" in demo_df.columns:
            demo_df["mfr_dt_num"] = pd.to_numeric(demo_df["mfr_dt"], errors="coerce").fillna(0)
        else:
            demo_df["mfr_dt_num"] = 0
        demo_df["primaryid_num"] = pd.to_numeric(demo_df["primaryid"], errors="coerce").fillna(0)

        deduped = (
            demo_df.sort_values(
                ["caseid", "caseversion_num", "mfr_dt_num", "primaryid_num"],
                ascending=[True, False, False, False],
            )
            .drop_duplicates(subset=["caseid"], keep="first")
            .drop(columns=["caseversion_num", "mfr_dt_num", "primaryid_num"])
            .reset_index(drop=True)
        )

        report.unique_cases_after = len(deduped)
        report.duplicate_rows_removed = report.total_rows_before - report.unique_cases_after
        report.duplicate_rate = (
            report.duplicate_rows_removed / report.total_rows_before
            if report.total_rows_before > 0 else 0.0
        )
        return deduped, report

    def _filter_to_valid_cases(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Keep only rows whose primaryid appears in the deduplicated DEMO table."""
        if self._valid_primaryids is None:
            raise ParseError("DEMO must be parsed before other FAERS files.")
        before = len(df)
        filtered = df[df["primaryid"].astype(str).isin(self._valid_primaryids)].copy()
        filtered = filtered.reset_index(drop=True)
        removed = before - len(filtered)
        if removed:
            logger.debug("%s: removed %d rows for duplicate/non-current primaryids.", file_type, removed)
        return filtered

    def _save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        """Save DataFrame as Parquet."""
        out = self._out_dir / f"{name}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        logger.info("  Saved %s -> %s (%d rows)", name, out.name, len(df))
        return out
