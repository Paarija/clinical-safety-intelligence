"""
common/paths.py

Central path resolver.  All data paths are derived from a single DATA_DIR
root so no module hardcodes absolute paths.

Usage:
    from clinical_safety.common.paths import Paths
    paths = Paths()
    faers_zip = paths.raw_faers / "faers_ascii_2026q1.zip"
"""

from __future__ import annotations

import os
from pathlib import Path

import clinical_safety.common.config  # noqa: F401


class Paths:
    """
    Resolves all project data and output paths from a single root.

    Priority order for DATA_DIR:
      1. DATA_DIR environment variable
      2. config value passed at construction
      3. ./data  (relative to current working directory)
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        env_dir = os.getenv("DATA_DIR")
        resolved = env_dir or data_dir or "data"
        self._data = Path(resolved).resolve()

    # ── Top-level ─────────────────────────────────────────────────────────────

    @property
    def data(self) -> Path:
        return self._data

    # ── Raw source files ──────────────────────────────────────────────────────

    @property
    def raw(self) -> Path:
        return self._data / "raw"

    @property
    def raw_faers(self) -> Path:
        return self.raw / "faers"

    @property
    def raw_clinicaltrials(self) -> Path:
        return self.raw / "clinicaltrials"

    @property
    def raw_fda(self) -> Path:
        return self.raw / "fda"

    @property
    def raw_pubmed(self) -> Path:
        return self.raw / "pubmed"

    # ── Interim (parsed / normalized / quality) ───────────────────────────────

    @property
    def interim(self) -> Path:
        return self._data / "interim"

    @property
    def interim_parsed(self) -> Path:
        return self.interim / "parsed"

    @property
    def interim_normalized(self) -> Path:
        return self.interim / "normalized"

    @property
    def interim_quality(self) -> Path:
        return self.interim / "quality_reports"

    # ── Processed (analytics / signals / evidence / reports) ─────────────────

    @property
    def processed(self) -> Path:
        return self._data / "processed"

    @property
    def processed_analytics(self) -> Path:
        return self.processed / "analytics"

    @property
    def processed_signals(self) -> Path:
        return self.processed / "signals"

    @property
    def processed_evidence(self) -> Path:
        return self.processed / "evidence_packets"

    @property
    def processed_reports(self) -> Path:
        return self.processed / "reports"

    @property
    def processed_checkpoints(self) -> Path:
        return self.processed / "checkpoints"

    # ── External reference data ───────────────────────────────────────────────

    @property
    def external(self) -> Path:
        return self._data / "external"

    @property
    def vocabularies(self) -> Path:
        return self.external / "vocabularies"

    @property
    def benchmark_sets(self) -> Path:
        return self.external / "benchmark_sets"

    # ── Reports ───────────────────────────────────────────────────────────────

    @property
    def reports(self) -> Path:
        return Path("reports").resolve()

    @property
    def case_studies(self) -> Path:
        return self.reports / "case_studies"

    @property
    def evaluation_reports(self) -> Path:
        return self.reports / "evaluation"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    # ── Source manifest ───────────────────────────────────────────────────────

    @property
    def source_manifest(self) -> Path:
        return self.interim / "source_manifest.json"

    # ── Utility ───────────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        """Create all directories if they do not exist."""
        dirs = [
            self.raw_faers,
            self.raw_clinicaltrials,
            self.raw_fda,
            self.raw_pubmed,
            self.interim_parsed,
            self.interim_normalized,
            self.interim_quality,
            self.processed_analytics,
            self.processed_signals,
            self.processed_evidence,
            self.processed_reports,
            self.processed_checkpoints,
            self.vocabularies,
            self.benchmark_sets,
            self.case_studies,
            self.evaluation_reports,
            self.figures,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def signal_evidence_dir(self, signal_id: str) -> Path:
        """Return (and create) a per-signal evidence directory."""
        p = self.processed_evidence / signal_id
        p.mkdir(parents=True, exist_ok=True)
        return p
