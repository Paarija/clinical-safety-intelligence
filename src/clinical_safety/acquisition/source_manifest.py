"""
acquisition/source_manifest.py

Tracks every raw source file ingested into the pipeline.
Every acquisition module must register its outputs here.
The manifest is serialized to data/interim/source_manifest.json.

Usage:
    from clinical_safety.acquisition.source_manifest import SourceManifest
    manifest = SourceManifest(paths)
    manifest.register(source_type="faers", path=zip_path, quarter="2026Q1")
    manifest.save()
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)


def _sha256(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class SourceManifest:
    """
    Records metadata about each raw source file loaded into the pipeline.

    Saved manifests contain:
      - metadata     : run-level provenance, including pipeline_run_id,
                       created_at, saved_at, and checksum_algorithm="sha256"
      - entries      : the existing list of source entries
      - summary      : count of entries per source_type

    Each entry contains:
      - source_type  : "faers" | "clinicaltrials" | "fda" | "pubmed"
      - path         : absolute path to raw file
      - size_bytes   : file size
      - sha256       : SHA-256 checksum when computed
      - registered_at: ISO timestamp
      - extra        : arbitrary source-specific metadata (quarter, query, etc.)
    """

    def __init__(self, paths: Paths | None = None) -> None:
        self._paths = paths or Paths()
        self._entries: list[dict[str, Any]] = []
        self._manifest_path = self._paths.source_manifest

        now = datetime.now(timezone.utc).isoformat()
        self._metadata: dict[str, Any] = {
            "schema_version": 2,
            "pipeline_run_id": str(uuid4()),
            "created_at": now,
            "checksum_algorithm": "sha256",
        }

        # Load existing manifests in either the legacy list shape or the
        # provenance-aware object shape. Run-level metadata intentionally stays
        # tied to this SourceManifest instance; loaded entries keep their own
        # original pipeline_run_id values.
        if self._manifest_path.exists():
            try:
                with self._manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._entries = data
                elif isinstance(data, dict):
                    entries = data.get("entries", [])
                    if isinstance(entries, list):
                        self._entries = entries
                    else:
                        logger.warning(
                            "Source manifest entries were not a list. Starting fresh."
                        )
                else:
                    logger.warning("Source manifest was not a list or object. Starting fresh.")
                for entry in self._entries:
                    if isinstance(entry, dict):
                        entry.setdefault("pipeline_run_id", "legacy-import")
                logger.debug("Loaded existing manifest with %d entries", len(self._entries))
            except Exception as exc:
                logger.warning("Could not load existing manifest: %s. Starting fresh.", exc)
                self._entries = []

    def register(
        self,
        source_type: str,
        path: str | Path,
        compute_checksum: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Register a source file in the manifest.

        Args:
            source_type     : Source category string.
            path            : Absolute or relative path to the raw file.
            compute_checksum: Whether to compute SHA-256 (skip for very large files).
            **extra         : Additional metadata (e.g. quarter="2026Q1").

        Returns:
            The manifest entry dict.
        """
        p = Path(path).resolve()

        entry: dict[str, Any] = {
            "source_type": source_type,
            "path": str(p),
            "file_name": p.name,
            "exists": p.exists(),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_run_id": self._metadata["pipeline_run_id"],
        }

        if p.exists():
            entry["size_bytes"] = p.stat().st_size
            if compute_checksum:
                entry["sha256"] = _sha256(p)
            else:
                entry["sha256"] = None
        else:
            entry["size_bytes"] = None
            entry["sha256"] = None
            logger.warning("Registering non-existent file: %s", p)

        entry.update(extra)

        # Replace existing entry for same path, or append
        self._entries = [e for e in self._entries if e.get("path") != str(p)]
        self._entries.append(entry)

        logger.info(
            "Manifest: registered %s source '%s' (%s bytes)",
            source_type,
            p.name,
            entry.get("size_bytes", "unknown"),
        )
        return entry

    def save(self) -> Path:
        """Write manifest to disk."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        self._metadata.setdefault("created_at", now)
        self._metadata.setdefault("checksum_algorithm", "sha256")
        self._metadata["saved_at"] = now
        payload = {
            "metadata": self._metadata,
            "entries": self._entries,
            "summary": self.summary(),
        }
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(
            "Source manifest saved to %s (%d entries)", self._manifest_path, len(self._entries)
        )
        return self._manifest_path

    def entries(self, source_type: str | None = None) -> list[dict[str, Any]]:
        """Return all entries, optionally filtered by source_type."""
        if source_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.get("source_type") == source_type]

    def metadata(self) -> dict[str, Any]:
        """Return run-level manifest provenance metadata."""
        return dict(self._metadata)

    def summary(self) -> dict[str, int]:
        """Return count of entries per source_type."""
        counts: dict[str, int] = {}
        for e in self._entries:
            st = e.get("source_type", "unknown")
            counts[st] = counts.get(st, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SourceManifest({self.summary()})"
