"""
Unit tests for drug and event normalization.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clinical_safety.normalization.drug_normalizer import DrugNormalizer
from clinical_safety.normalization.event_normalizer import EventNormalizer
from clinical_safety.common.types import MappingConfidence


class TestDrugNormalizer:
    """Tests for DrugNormalizer matching pipeline."""

    @pytest.fixture(autouse=True)
    def normalizer(self):
        self.norm = DrugNormalizer()

    def test_exact_match_normalized_name(self):
        drug_id, conf = self.norm.normalize("semaglutide")
        assert drug_id == "semaglutide"
        assert conf == MappingConfidence.EXACT

    def test_alias_match_brand_name(self):
        """Ozempic should match to semaglutide via alias."""
        drug_id, conf = self.norm.normalize("ozempic")
        assert drug_id == "semaglutide"
        assert conf == MappingConfidence.ALIAS

    def test_alias_match_case_insensitive(self):
        """Matching should be case-insensitive."""
        drug_id, conf = self.norm.normalize("OZEMPIC")
        assert drug_id == "semaglutide"
        assert conf == MappingConfidence.ALIAS

    def test_alias_match_wegovy(self):
        drug_id, conf = self.norm.normalize("WEGOVY")
        assert drug_id == "semaglutide"
        assert conf == MappingConfidence.ALIAS

    def test_alias_match_victoza(self):
        drug_id, conf = self.norm.normalize("VICTOZA")
        assert drug_id == "liraglutide"
        assert conf == MappingConfidence.ALIAS

    def test_alias_match_mounjaro(self):
        drug_id, conf = self.norm.normalize("MOUNJARO")
        assert drug_id == "tirzepatide"
        assert conf == MappingConfidence.ALIAS

    def test_unmatched_returns_none(self):
        """An unknown drug should return None with UNMATCHED confidence."""
        drug_id, conf = self.norm.normalize("completely_unknown_drug_xyz")
        assert drug_id is None
        assert conf == MappingConfidence.UNMATCHED

    def test_empty_string_unmatched(self):
        drug_id, conf = self.norm.normalize("")
        assert drug_id is None
        assert conf == MappingConfidence.UNMATCHED

    def test_normalize_series(self):
        series = pd.Series(["OZEMPIC", "VICTOZA", "UNKNOWN_DRUG"])
        drug_ids, confidences = self.norm.normalize_series(series)
        assert drug_ids.iloc[0] == "semaglutide"
        assert drug_ids.iloc[1] == "liraglutide"
        assert drug_ids.iloc[2] is None or pd.isna(drug_ids.iloc[2])

    def test_dosage_suffix_stripping(self):
        """'semaglutide 0.5 mg' should still match after stripping the dosage."""
        drug_id, conf = self.norm.normalize("semaglutide 0.5 mg")
        assert drug_id == "semaglutide"

    def test_audit_is_populated(self):
        self.norm.normalize("OZEMPIC")
        self.norm.normalize("UNKNOWN_X")
        summary = self.norm.confidence_summary()
        assert summary[MappingConfidence.ALIAS.value] >= 1
        assert summary[MappingConfidence.UNMATCHED.value] >= 1

    def test_validate_mapping_coverage_uses_unique_audit_entries(self):
        self.norm.normalize("OZEMPIC")
        self.norm.normalize("OZEMPIC")
        self.norm.normalize("UNKNOWN_X")

        coverage = self.norm.validate_mapping_coverage()

        assert coverage == {
            "total": 2,
            "matched": 1,
            "coverage_pct": 50.0,
            "threshold_pct": 90.0,
            "passed": False,
        }

    def test_validate_mapping_coverage_passes_custom_threshold(self):
        self.norm.normalize("OZEMPIC")
        self.norm.normalize("UNKNOWN_X")

        coverage = self.norm.validate_mapping_coverage(threshold=0.50)

        assert coverage["passed"] is True
        assert coverage["threshold_pct"] == 50.0


class TestEventNormalizer:
    """Tests for EventNormalizer matching pipeline."""

    @pytest.fixture(autouse=True)
    def normalizer(self):
        self.norm = EventNormalizer()

    def test_exact_match_pancreatitis(self):
        event_id, conf = self.norm.normalize("pancreatitis")
        assert event_id == "pancreatitis"
        assert conf == MappingConfidence.EXACT

    def test_exact_match_pancreatitis_acute(self):
        event_id, conf = self.norm.normalize("pancreatitis acute")
        assert event_id == "pancreatitis"
        assert conf == MappingConfidence.EXACT

    def test_exact_match_case_insensitive(self):
        event_id, conf = self.norm.normalize("Pancreatitis Acute")
        assert event_id == "pancreatitis"
        assert conf == MappingConfidence.EXACT

    def test_cholelithiasis_maps_to_gallbladder(self):
        event_id, conf = self.norm.normalize("cholelithiasis")
        assert event_id == "gallbladder_disease"

    def test_cholecystitis_acute_maps_to_gallbladder(self):
        event_id, conf = self.norm.normalize("cholecystitis acute")
        assert event_id == "gallbladder_disease"

    def test_vomiting_maps_to_nausea_family(self):
        event_id, conf = self.norm.normalize("vomiting")
        assert event_id == "severe_nausea_vomiting"

    def test_dehydration_maps_correctly(self):
        event_id, conf = self.norm.normalize("dehydration")
        assert event_id == "dehydration"

    def test_unmatched_event(self):
        event_id, conf = self.norm.normalize("sporadic_intergalactic_syndrome")
        assert event_id is None
        assert conf == MappingConfidence.UNMATCHED

    def test_normalize_series(self):
        series = pd.Series(["pancreatitis acute", "cholelithiasis", "unknown_event_xyz"])
        event_ids, _ = self.norm.normalize_series(series)
        assert event_ids.iloc[0] == "pancreatitis"
        assert event_ids.iloc[1] == "gallbladder_disease"
        assert event_ids.iloc[2] is None or pd.isna(event_ids.iloc[2])

    def test_validate_mapping_coverage_uses_unique_audit_entries(self):
        self.norm.normalize("pancreatitis")
        self.norm.normalize("pancreatitis")
        self.norm.normalize("sporadic_intergalactic_syndrome")

        coverage = self.norm.validate_mapping_coverage()

        assert coverage == {
            "total": 2,
            "matched": 1,
            "coverage_pct": 50.0,
            "threshold_pct": 90.0,
            "passed": False,
        }
