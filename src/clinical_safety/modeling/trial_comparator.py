"""
modeling/trial_comparator.py

Parses raw ClinicalTrials.gov JSON files and extracts arm-level
adverse-event rates for each drug-event pair.

Responsibilities:
  - Load raw JSON files from data/raw/clinicaltrials/
  - Extract arm group metadata (treatment vs. comparator/placebo)
  - Extract serious-event and other-event tables per arm
  - Compute event_rate = affected / at_risk for each event-arm pair
  - Normalize event terms to in-scope event_ids (EventNormalizer)
  - Compute absolute_risk_difference (treatment_rate - comparator_rate)
  - Save trial_comparison.parquet to data/processed/analytics/

Important caveats:
  - Event rates are from trial arms, NOT from FAERS reports.
  - Missing at_risk values are recorded as NaN, not imputed.
  - Unavailability is explicit — never silently dropped.

Usage:
    from clinical_safety.modeling.trial_comparator import TrialComparator
    tc = TrialComparator()
    df = tc.compare()   # returns trial_comparison DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from clinical_safety.acquisition.clinicaltrials_types import (
    ClinicalTrialsRawFileEnvelope,
    ClinicalTrialsStudy,
    format_validation_error,
)
from clinical_safety.common.exceptions import DataSourceError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.normalization.event_normalizer import EventNormalizer

logger = get_logger(__name__)

# Arm type keywords (case-insensitive substring match)
_COMPARATOR_KEYWORDS = ("placebo", "comparator", "control", "vehicle")
_TREATMENT_KEYWORDS = ("experimental", "active", "treatment")


def _detect_arm_type(label: str, arm_type_str: str | None) -> str:
    """
    Classify a trial arm as 'treatment', 'comparator', or 'unknown'.

    ClinicalTrials v2 reports ArmGroupType as: EXPERIMENTAL, ACTIVE_COMPARATOR,
    PLACEBO_COMPARATOR, etc.  We normalize these.
    """
    joined = (str(label) + " " + str(arm_type_str or "")).lower()
    if any(k in joined for k in _COMPARATOR_KEYWORDS):
        return "comparator"
    if any(k in joined for k in _TREATMENT_KEYWORDS):
        return "treatment"
    return "unknown"


def _safe_int(val: Any) -> int | None:
    """Convert to int safely; return None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_rate(affected: int | None, at_risk: int | None) -> float | None:
    """Compute affected / at_risk safely."""
    if affected is None or at_risk is None or at_risk == 0:
        return None
    return round(affected / at_risk, 6)


class TrialComparator:
    """
    Extracts arm-level adverse event rates from raw ClinicalTrials JSON
    and produces a comparison table joinable with FAERS signals.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        self._paths = paths or Paths()
        self._raw_dir = self._paths.raw_clinicaltrials
        self._out_dir = self._paths.processed_analytics
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._event_normalizer = EventNormalizer(paths=self._paths)

    def compare(self, save: bool = True) -> pd.DataFrame:
        """
        Load all raw ClinicalTrials JSON files and produce comparison table.

        Returns:
            DataFrame with columns:
                drug_id, nct_id, arm_label, arm_type,
                event_term, event_id, is_serious,
                affected, at_risk, event_rate,
                comparator_rate, absolute_risk_difference
        """
        json_files = list(self._raw_dir.glob("*.json"))
        if not json_files:
            logger.warning(
                "No ClinicalTrials JSON files found in %s. "
                "Run ClinicalTrialsSource.acquire() first.",
                self._raw_dir,
            )
            return pd.DataFrame()

        all_rows: list[dict[str, Any]] = []
        for jf in json_files:
            try:
                rows = self._process_drug_file(jf)
                all_rows.extend(rows)
            except Exception as exc:
                logger.warning("Failed to process %s: %s — skipping.", jf.name, exc)

        if not all_rows:
            logger.warning("No trial AE rows extracted from any file.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = self._add_risk_difference(df)

        if save:
            out = self._out_dir / "trial_comparison.parquet"
            df.to_parquet(out, index=False)
            logger.info("Trial comparison saved -> %s (%d rows)", out.name, len(df))
            # Also CSV for human inspection
            df.to_csv(self._out_dir / "trial_comparison.csv", index=False)

        logger.info(
            "Trial comparator: %d total AE rows from %d drug files",
            len(df), len(json_files),
        )
        return df

    def _process_drug_file(self, path: Path) -> list[dict[str, Any]]:
        """Process one drug's JSON file into a list of AE row dicts."""
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DataSourceError(
                f"ClinicalTrials raw file {path.name} is not valid JSON: {exc.msg}"
            ) from exc

        try:
            raw_file = ClinicalTrialsRawFileEnvelope.model_validate(data)
        except ValidationError as exc:
            raise DataSourceError(
                f"ClinicalTrials raw file {path.name} schema mismatch: expected an object "
                f"with a required 'studies' list; {format_validation_error(exc)}"
            ) from exc

        drug_id = raw_file.drug_id or path.stem
        studies = raw_file.studies
        rows: list[dict[str, Any]] = []

        for study_index, study in enumerate(studies):
            validated_study = self._validate_study(study, drug_id, path, study_index)
            if validated_study is None:
                continue
            study_data, nct_id = validated_study
            arm_map = self._extract_arms(study_data, nct_id)
            study_rows = self._extract_ae_rows(study_data, drug_id, nct_id, arm_map)
            rows.extend(study_rows)

        logger.debug("  %s: %d studies -> %d AE rows", path.name, len(studies), len(rows))
        return rows

    def _validate_study(
        self,
        study: dict[str, Any],
        drug_id: str,
        path: Path,
        study_index: int,
    ) -> tuple[dict[str, Any], str] | None:
        """Validate one saved raw study and return its dict plus NCT id."""
        try:
            parsed = ClinicalTrialsStudy.model_validate(study)
        except ValidationError as exc:
            nct_hint = self._extract_nct_id(study) if isinstance(study, dict) else None
            nct_context = f", nct_id={nct_hint}" if nct_hint else ""
            logger.warning(
                "Skipping malformed ClinicalTrials study in %s for drug_id=%s at studies[%d]%s: %s",
                path.name,
                drug_id,
                study_index,
                nct_context,
                format_validation_error(exc),
            )
            return None
        return parsed.model_dump(mode="python"), parsed.nct_id

    @staticmethod
    def _extract_nct_id(study: dict) -> str | None:
        """Pull NCT ID from the v2 study structure."""
        # v2 wraps everything under protocolSection / resultsSection
        try:
            return study["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _extract_arms(study: dict, nct_id: str) -> dict[str, str]:
        """
        Build a mapping of {arm_group_id -> arm_type} from study arms.

        ClinicalTrials v2 stores arm groups in:
          protocolSection.armsInterventionsModule.armGroups[]
        Each has: label, type (EXPERIMENTAL|ACTIVE_COMPARATOR|PLACEBO_COMPARATOR|OTHER)
        """
        arm_map: dict[str, str] = {}
        try:
            arms = (
                study.get("protocolSection", {})
                     .get("armsInterventionsModule", {})
                     .get("armGroups", [])
            )
            for arm in arms:
                label = arm.get("label", "")
                arm_type_str = arm.get("type", "")
                arm_map[label] = _detect_arm_type(label, arm_type_str)
        except Exception as exc:
            logger.debug("  %s: arm extraction error: %s", nct_id, exc)
        return arm_map

    def _extract_ae_rows(
        self,
        study: dict,
        drug_id: str,
        nct_id: str,
        arm_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Extract all serious and other AE rows from resultsSection.
        ClinicalTrials v2 stores events in:
          resultsSection.adverseEventsModule.seriousEvents[] and otherEvents[]
        Each event has: term, organSystem, stats[] (one per arm group)
        """
        rows: list[dict[str, Any]] = []
        try:
            ae_module = (
                study.get("resultsSection", {})
                     .get("adverseEventsModule", {})
            )
            # event_groups maps groupId -> title
            event_groups = {
                g["id"]: g.get("title", g["id"])
                for g in ae_module.get("eventGroups", [])
            }

            for is_serious, event_list_key in [(True, "seriousEvents"), (False, "otherEvents")]:
                for event in ae_module.get(event_list_key, []):
                    term = event.get("term", "unknown")
                    event_id, _ = self._event_normalizer.normalize(term)

                    for stat in event.get("stats", []):
                        group_id = stat.get("groupId", "")
                        group_label = event_groups.get(group_id, group_id)
                        arm_type = arm_map.get(group_label, _detect_arm_type(group_label, None))
                        affected = _safe_int(stat.get("numAffected"))
                        at_risk = _safe_int(stat.get("numAtRisk"))

                        rows.append({
                            "drug_id": drug_id,
                            "nct_id": nct_id,
                            "arm_label": group_label,
                            "arm_type": arm_type,
                            "event_term": term,
                            "event_id": event_id,
                            "is_serious": is_serious,
                            "affected": affected,
                            "at_risk": at_risk,
                            "event_rate": _safe_rate(affected, at_risk),
                        })
        except Exception as exc:
            logger.warning(
                "Skipping ClinicalTrials AE extraction for drug_id=%s nct_id=%s: %s",
                drug_id,
                nct_id,
                exc,
            )
        return rows

    @staticmethod
    def _add_risk_difference(df: pd.DataFrame) -> pd.DataFrame:
        """
        For each (drug_id, nct_id, event_id) pair, compute the
        absolute_risk_difference = treatment_rate - comparator_rate.

        This is done cross-arm within each study.
        Missing values are left as NaN — never imputed.
        """
        if df.empty:
            return df

        # Get treatment and comparator rates per study+event pair
        treatment = df[df["arm_type"] == "treatment"].groupby(
            ["drug_id", "nct_id", "event_id"]
        )["event_rate"].mean().rename("treatment_rate")

        comparator = df[df["arm_type"] == "comparator"].groupby(
            ["drug_id", "nct_id", "event_id"]
        )["event_rate"].mean().rename("comparator_rate")

        rate_df = pd.concat([treatment, comparator], axis=1).reset_index()
        rate_df["absolute_risk_difference"] = (
            rate_df["treatment_rate"] - rate_df["comparator_rate"]
        ).round(6)

        # Merge back onto the full AE table
        df = df.merge(
            rate_df[["drug_id", "nct_id", "event_id",
                     "treatment_rate", "comparator_rate", "absolute_risk_difference"]],
            on=["drug_id", "nct_id", "event_id"],
            how="left",
        )
        return df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        tc = TrialComparator()
        df = tc.compare()
        print(f"\nTrial comparison: {len(df)} rows, {df['nct_id'].nunique() if not df.empty else 0} studies")
        if not df.empty:
            print(df[["drug_id", "nct_id", "event_term", "arm_type", "event_rate"]].head(20))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"Trial comparison failed: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
