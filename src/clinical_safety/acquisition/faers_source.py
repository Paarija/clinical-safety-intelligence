"""
acquisition/faers_source.py

FAERS/AEMS quarterly public data acquisition.

Responsibilities:
  - Locate the FAERS quarterly ZIP file (from config or explicit path)
  - Validate that required files are present inside the ZIP
  - Extract files to data/raw/faers/extracted/
  - Register in source manifest

Does NOT parse, clean, or modify the files in any way.

Usage:
    from clinical_safety.acquisition.faers_source import FAERSSource
    src = FAERSSource()
    paths = src.acquire()   # returns dict of file type -> Path
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from clinical_safety.acquisition.source_manifest import SourceManifest
from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import DataSourceError, SourceFileNotFoundError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)
_USER_AGENT = "clinical-safety-intelligence/0.1.0 (research pipeline)"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# Required file types (FAERS file prefix → table name)
REQUIRED_FILE_TYPES = {
    "DEMO": "demographics",
    "DRUG": "drug",
    "REAC": "reactions",
    "OUTC": "outcomes",
}

OPTIONAL_FILE_TYPES = {
    "INDI": "indications",
    "THER": "therapy_dates",
    "RPSR": "report_source",
}


class FAERSSource:
    """
    Acquires FAERS quarterly ASCII public data from a ZIP file.

    The ZIP is expected at data_sources.yaml faers.zip_path. If that configured
    file is missing and faers.auto_download is enabled, the upstream ZIP is
    downloaded first. Files are extracted to data/raw/faers/extracted/.
    """

    def __init__(
        self,
        zip_path: str | Path | None = None,
        paths: Paths | None = None,
        manifest: SourceManifest | None = None,
    ) -> None:
        cfg = get_config()
        self._cfg = cfg.data_sources.faers
        self._paths = paths or Paths()
        self._manifest = manifest or SourceManifest(self._paths)

        self._explicit_zip_path = zip_path is not None
        configured_zip = Path(zip_path or self._cfg.zip_path)
        if configured_zip.is_absolute():
            self._zip_path = configured_zip
        elif configured_zip.parts and configured_zip.parts[0] == "data":
            self._zip_path = self._paths.data.joinpath(*configured_zip.parts[1:])
        else:
            self._zip_path = Path(__file__).resolve().parents[3] / configured_zip

        self._extract_dir = self._paths.raw_faers / "extracted"
        self._quarter = self._cfg.quarter

    def acquire(self) -> dict[str, Path]:
        """
        Validate, extract, and register FAERS quarterly files.

        Returns:
            Dict mapping file type key (e.g. "DEMO") to extracted file Path.

        Raises:
            SourceFileNotFoundError: ZIP not found.
            DataSourceError        : Required files missing from ZIP.
        """
        logger.info("FAERS acquisition starting — ZIP: %s", self._zip_path)

        if not self._zip_path.exists():
            self._ensure_zip_available()

        # Register the ZIP itself in the manifest
        self._manifest.register(
            source_type="faers",
            path=self._zip_path,
            quarter=self._quarter,
            file_type="zip_archive",
        )

        # Extract
        self._extract_dir.mkdir(parents=True, exist_ok=True)
        extracted = self._extract_zip()

        # Validate required files exist
        found: dict[str, Path] = {}
        missing: list[str] = []

        for prefix, table_name in REQUIRED_FILE_TYPES.items():
            match = self._find_file(extracted, prefix)
            if match:
                found[prefix] = match
                self._manifest.register(
                    source_type="faers",
                    path=match,
                    quarter=self._quarter,
                    file_type=table_name,
                    compute_checksum=False,  # large files; skip for speed
                )
                logger.info("  Found required file: %s -> %s", prefix, match.name)
            else:
                missing.append(prefix)

        if missing:
            raise DataSourceError(
                f"FAERS ZIP is missing required file(s) with prefix(es): {missing}. "
                f"Files found in ZIP: {[p.name for p in extracted]}"
            )

        # Register optional files (no error if missing)
        for prefix, table_name in OPTIONAL_FILE_TYPES.items():
            match = self._find_file(extracted, prefix)
            if match:
                found[prefix] = match
                self._manifest.register(
                    source_type="faers",
                    path=match,
                    quarter=self._quarter,
                    file_type=table_name,
                    compute_checksum=False,
                )
                logger.info("  Found optional file: %s -> %s", prefix, match.name)
            else:
                logger.debug("  Optional file not found: %s (skipping)", prefix)

        self._manifest.save()
        logger.info(
            "FAERS acquisition complete. %d required + %d optional files ready.",
            len(REQUIRED_FILE_TYPES),
            len(found) - len(REQUIRED_FILE_TYPES),
        )
        return found

    def _ensure_zip_available(self) -> None:
        """Download the configured FAERS ZIP when allowed, or raise a clear missing-file error."""
        if self._explicit_zip_path:
            raise SourceFileNotFoundError(
                f"FAERS ZIP not found at explicit path: {self._zip_path}\n"
                "Explicit zip_path overrides are treated as local files and are not auto-downloaded. "
                "Provide an existing ZIP path or remove the override to use faers.auto_download."
            )

        if not self._cfg.auto_download or not self._cfg.download_url:
            raise SourceFileNotFoundError(
                f"FAERS ZIP not found at: {self._zip_path}\n"
                f"Place faers_ascii_{self._quarter.lower()}.zip in {self._paths.raw_faers}, "
                "or set faers.download_url and faers.auto_download=true in configs/data_sources.yaml."
            )

        self._download_zip()

    def _download_zip(self) -> None:
        """Stream the configured FAERS ZIP to a .part file, validate it, then atomically replace."""
        url = self._cfg.download_url.strip()
        part_path = Path(f"{self._zip_path}.part")
        bytes_written = 0
        response = None

        logger.info("FAERS ZIP missing; downloading from %s to %s", url, self._zip_path)
        try:
            self._zip_path.parent.mkdir(parents=True, exist_ok=True)
            if part_path.exists():
                part_path.unlink()
            response = requests.get(
                url,
                stream=True,
                timeout=(
                    self._cfg.request_connect_timeout_sec,
                    self._cfg.request_read_timeout_sec,
                ),
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
            with part_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
        except requests.exceptions.HTTPError as exc:
            self._remove_partial_download(part_path)
            error_response = getattr(exc, "response", None)
            status_code = error_response.status_code if error_response is not None else "unknown"
            raise DataSourceError(
                f"FAERS download failed with HTTP {status_code} for {url}. "
                f"Download the ZIP manually to {self._zip_path} or update faers.download_url."
            ) from exc
        except requests.exceptions.RequestException as exc:
            self._remove_partial_download(part_path)
            raise DataSourceError(
                f"FAERS download failed for {url}: {exc}. "
                f"Download the ZIP manually to {self._zip_path} or retry when the FDA export is reachable."
            ) from exc
        except OSError as exc:
            self._remove_partial_download(part_path)
            raise DataSourceError(
                f"FAERS download could not be written to {self._zip_path}: {exc}"
            ) from exc
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

        if bytes_written == 0 or not part_path.exists() or part_path.stat().st_size == 0:
            self._remove_partial_download(part_path)
            raise DataSourceError(
                f"FAERS download from {url} produced an empty file. "
                f"Download the ZIP manually to {self._zip_path} or retry with a valid faers.download_url."
            )

        if not zipfile.is_zipfile(part_path):
            self._remove_partial_download(part_path)
            raise DataSourceError(
                f"FAERS download from {url} did not produce a valid ZIP archive. "
                f"Download the ZIP manually to {self._zip_path} or retry with a valid faers.download_url."
            )

        try:
            part_path.replace(self._zip_path)
        except OSError as exc:
            self._remove_partial_download(part_path)
            raise DataSourceError(
                f"FAERS download could not be moved into place at {self._zip_path}: {exc}"
            ) from exc
        logger.info("FAERS ZIP downloaded to %s (%d bytes)", self._zip_path, bytes_written)

    @staticmethod
    def _remove_partial_download(part_path: Path) -> None:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove partial FAERS download: %s", part_path)

    def _extract_zip(self) -> list[Path]:
        """Extract all files from the ZIP. Returns list of extracted Paths."""
        extracted: list[Path] = []
        try:
            with zipfile.ZipFile(self._zip_path, "r") as zf:
                members = zf.infolist()
                logger.info("  ZIP contains %d entries. Extracting to %s ...", len(members), self._extract_dir)
                for member in members:
                    # Flatten nested directories inside some FAERS ZIPs
                    member_path = Path(member.filename)
                    target = self._extract_dir / member_path.name
                    if not target.exists() or target.stat().st_size == 0:
                        with zf.open(member) as src, target.open("wb") as dst:
                            dst.write(src.read())
                    extracted.append(target)
        except zipfile.BadZipFile as exc:
            raise DataSourceError(f"FAERS ZIP is corrupted or not a valid ZIP file: {exc}") from exc
        return extracted

    @staticmethod
    def _find_file(candidates: list[Path], prefix: str) -> Path | None:
        """Find the first file whose name starts with the given prefix (case-insensitive)."""
        prefix_upper = prefix.upper()
        for p in candidates:
            if p.name.upper().startswith(prefix_upper) and p.suffix.upper() in (".TXT", ".CSV"):
                return p
        return None


if __name__ == "__main__":
    import sys
    
    try:
        from clinical_safety.common.logging import get_logger
        logging_mod = __import__("clinical_safety.common.logging", fromlist=["setup_logging"])
        if hasattr(logging_mod, "setup_logging"):
            logging_mod.setup_logging()
    except (ImportError, AttributeError):
        pass

    try:
        # 1. Ingestion / Acquisition
        src = FAERSSource()
        file_map = src.acquire()

        # 2. Parsing & Deduplication
        from clinical_safety.parsing.faers_parser import FAERSParser
        parser = FAERSParser()
        tables = parser.parse_all(file_map)
        dedup_report = parser.dedup_report

        # 3. Normalization
        from clinical_safety.normalization.drug_normalizer import DrugNormalizer
        from clinical_safety.normalization.event_normalizer import EventNormalizer
        from clinical_safety.normalization.outcome_normalizer import OutcomeNormalizer

        drug_norm = DrugNormalizer()
        drug_norm_df = drug_norm.apply_to_dataframe(tables["DRUG"])
        drug_norm.save_audit()

        event_norm = EventNormalizer()
        reac_norm_df = event_norm.apply_to_dataframe(tables["REAC"])
        event_norm.save_audit()

        outc_norm = OutcomeNormalizer()
        outc_norm_df = outc_norm.apply_to_dataframe(tables["OUTC"])

        # Save normalized to paths.interim_normalized
        paths = Paths()
        paths.interim_normalized.mkdir(parents=True, exist_ok=True)
        tables["DEMO"].to_parquet(paths.interim_normalized / "faers_demo_normalized.parquet", index=False)
        drug_norm_df.to_parquet(paths.interim_normalized / "faers_drug_normalized.parquet", index=False)
        reac_norm_df.to_parquet(paths.interim_normalized / "faers_reac_normalized.parquet", index=False)
        outc_norm_df.to_parquet(paths.interim_normalized / "faers_outc_normalized.parquet", index=False)

        # 4. Data Quality Report
        from clinical_safety.quality.data_quality import DataQualityReporter
        reporter = DataQualityReporter()
        norm_tables = {
            "DEMO": tables["DEMO"],
            "DRUG": drug_norm_df,
            "REAC": reac_norm_df,
            "OUTC": outc_norm_df,
        }
        report = reporter.run(
            tables=norm_tables,
            dedup_report=dedup_report,
            drug_confidence_summary=drug_norm.confidence_summary(),
            event_confidence_summary=event_norm.confidence_summary()
        )
        reporter.save(report)

        print("\nFAERS Data Ingestion Pipeline completed successfully!")
        sys.exit(0)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        import traceback
        print(f"\nFAERS Pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
