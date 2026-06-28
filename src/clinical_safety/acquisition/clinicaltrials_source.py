from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

from clinical_safety.acquisition.clinicaltrials_types import (
    ClinicalTrialsResponseEnvelope,
    format_validation_error,
)
from clinical_safety.acquisition.source_manifest import SourceManifest
from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import DataSourceError, EvidenceRetrievalError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths

logger = get_logger(__name__)

# ClinicalTrials v2 API endpoint
_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_USER_AGENT = "clinical-safety-intelligence/0.1.0 (research pipeline)"


class ClinicalTrialsSource:
    """
    Acquires ClinicalTrials.gov study data for all in-scope GLP-1 drugs.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        self._paths = paths or Paths()
        cfg = get_config()
        self._ct_cfg = cfg.data_sources.clinicaltrials
        self._drug_cfg = cfg.drug_scope
        self._manifest = SourceManifest(self._paths)
        self._delay = self._ct_cfg.request_delay_sec

    def acquire(self) -> dict[str, Path]:
        """Acquire raw JSON for all configured drugs.

        Returns:
            dict mapping drug_id -> saved raw JSON Path.
        """
        results: dict[str, Path] = {}
        for drug_entry in self._drug_cfg.drugs:
            try:
                path = self._acquire_drug(drug_entry.id, drug_entry.normalized_name)
                results[drug_entry.id] = path
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Log but do not abort — missing data is marked unavailable downstream
                logger.warning(
                    "ClinicalTrials acquisition failed for %s: %s — continuing.",
                    drug_entry.id, exc,
                )

        self._manifest.save()
        return results

    def _acquire_drug(self, drug_id: str, drug_name: str) -> Path:
        """Acquire all pages for one drug and write a raw JSON file."""
        out_dir = self._paths.raw_clinicaltrials
        out_dir.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {
            "query.intr": drug_name,
            "filter.overallStatus": "COMPLETED",
            "pageSize": self._ct_cfg.page_size,
        }

        all_studies: list[dict[str, Any]] = []
        page_token: str | None = None
        page_num = 0

        while page_num < self._ct_cfg.max_pages:
            params = self._build_params(drug_name, page_token)
            response_data = self._get_page(params)

            studies = response_data.get("studies", [])
            filtered_studies = [
                study for study in studies
                if isinstance(study.get("resultsSection"), dict)
                and study["resultsSection"].get("adverseEventsModule")
            ]
            all_studies.extend(filtered_studies)

            next_token = response_data.get("nextPageToken")
            total_count = response_data.get("totalCount", 0)

            logger.debug(
                "  Page %d: %d studies retrieved (filtered to %d with results, total_count=%d, has_next=%s)",
                page_num + 1, len(studies), len(filtered_studies), total_count, bool(next_token),
            )

            if not next_token or not studies:
                break

            page_token = next_token
            page_num += 1
            time.sleep(self._delay)

        out_path = out_dir / f"{drug_id}.json"
        out_path.write_text(json.dumps({"studies": all_studies}, indent=2), encoding="utf-8")
        self._manifest.register(source_type="clinicaltrials", path=out_path, drug_id=drug_id)
        return out_path

    def _build_params(self, drug_name: str, page_token: str | None) -> dict[str, Any]:
        """Build the API query parameter dict for one page."""
        params: dict[str, Any] = {
            "query.intr": drug_name,
            "filter.overallStatus": "COMPLETED",
            "pageSize": self._ct_cfg.page_size,
        }
        if page_token:
            params["pageToken"] = page_token
        return params

    def _get_page(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch one page of studies from the ClinicalTrials.gov API."""
        try:
            resp = requests.get(
                _BASE_URL,
                params=params,
                timeout=(self._ct_cfg.request_connect_timeout_sec, self._ct_cfg.request_read_timeout_sec),
                headers={
                    "Accept": "application/json",
                    "User-Agent": _USER_AGENT,
                },
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            raise EvidenceRetrievalError(
                f"ClinicalTrials API HTTP error: {status_code} — {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise EvidenceRetrievalError(f"ClinicalTrials API fetch failed: {exc}") from exc

        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataSourceError("ClinicalTrials API response was not valid JSON") from exc

        try:
            envelope = ClinicalTrialsResponseEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise DataSourceError(
                "ClinicalTrials API response schema mismatch for "
                f"query.intr={params.get('query.intr', '<unknown>')!r}: expected an object "
                f"with a required 'studies' list; {format_validation_error(exc)}"
            ) from exc

        return envelope.model_dump(mode="python")


def main() -> int:
    """Run ClinicalTrials.gov acquisition as a CLI entrypoint."""
    try:
        source = ClinicalTrialsSource()
        expected_count = len(source._drug_cfg.drugs)
        results = source.acquire()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"ClinicalTrials acquisition failed: {exc}", file=sys.stderr)
        return 1

    acquired_count = len(results)
    summary = f"ClinicalTrials acquisition complete: {acquired_count}/{expected_count} drugs acquired."
    if acquired_count < expected_count:
        print(summary, file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
