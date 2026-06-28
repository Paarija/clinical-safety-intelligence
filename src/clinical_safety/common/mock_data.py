"""Helpers for detecting synthetic pipeline outputs."""

from __future__ import annotations

from pathlib import Path

from clinical_safety.common.paths import Paths


def is_mock_pipeline_data(paths: Paths | None = None) -> bool:
    """Return True while any mock-generated pipeline artifact is still current."""
    paths = paths or Paths()
    marker = paths.processed_analytics / "_IS_MOCK_DATA"
    if not marker.exists():
        return False

    marker_mtime = marker.stat().st_mtime
    artifacts: list[Path] = [
        paths.processed_signals / "candidate_signals.parquet",
        paths.processed_analytics / "trial_comparison.parquet",
        paths.interim_quality / "data_quality_report.json",
        paths.source_manifest,
    ]
    if any(not artifact.exists() for artifact in artifacts):
        return True

    if not all(artifact.stat().st_mtime > marker_mtime for artifact in artifacts):
        return True

    derived_dirs = [paths.processed_evidence, paths.processed_reports]
    for derived_dir in derived_dirs:
        if not derived_dir.exists():
            continue
        has_stale_derived_output = any(
            path.stat().st_mtime <= marker_mtime
            for path in derived_dir.rglob("*")
            if path.is_file()
        )
        if has_stale_derived_output:
            return True

    return False
