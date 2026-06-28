"""
Unit tests for signal metric computation (ROR, PRR, CI).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from clinical_safety.analytics.disproportionality import DisproportionalityCalculator


class TestRORCalculation:
    """
    ROR = (a * d) / (b * c)
    For a perfect signal: large a, small b, small c, large d
    For null signal: a/b == c/d  =>  ROR ~= 1.0
    """

    @pytest.fixture(autouse=True)
    def calc(self):
        self.calc = DisproportionalityCalculator()

    def _make_ct(self, a, b, c, d) -> pd.DataFrame:
        return pd.DataFrame([{
            "drug_id": "test_drug",
            "event_id": "test_event",
            "a": a, "b": b, "c": c, "d": d,
        }])

    def test_ror_strong_signal(self):
        """
        a=100 drug+event, b=10 drug-noevent, c=5 nodrug+event, d=5000 nodrug-noevent
        ROR = (100*5000)/(10*5) = 10000 => very strong signal
        """
        df = self._make_ct(100, 10, 5, 5000)
        result = self.calc.compute(df)
        assert result.iloc[0]["ror"] > 100

    def test_ror_null_signal(self):
        """When a/b == c/d, ROR should be ~1.0."""
        df = self._make_ct(50, 50, 100, 100)
        result = self.calc.compute(df)
        assert abs(result.iloc[0]["ror"] - 1.0) < 0.05

    def test_ci_direction(self):
        """Lower CI must be <= ROR <= Upper CI."""
        df = self._make_ct(20, 80, 10, 500)
        result = self.calc.compute(df)
        row = result.iloc[0]
        assert row["ror_lower_ci"] <= row["ror"]
        assert row["ror"] <= row["ror_upper_ci"]

    def test_strong_signal_ci_above_1(self):
        """For a strong signal, even the lower CI should be > 1.0."""
        df = self._make_ct(50, 10, 20, 3000)
        result = self.calc.compute(df)
        assert result.iloc[0]["ror_lower_ci"] > 1.0

    def test_prr_computed(self):
        """PRR should be positive and non-NaN for a valid table."""
        df = self._make_ct(50, 50, 100, 1000)
        result = self.calc.compute(df)
        assert not math.isnan(result.iloc[0]["prr"])
        assert result.iloc[0]["prr"] > 0

    def test_zero_cell_handling(self):
        """Zero cells should not cause division by zero (Haldane correction applied)."""
        df = self._make_ct(5, 0, 20, 1000)  # b=0
        result = self.calc.compute(df)
        assert not math.isnan(result.iloc[0]["ror"])
        assert not math.isnan(result.iloc[0]["ror_lower_ci"])

    def test_signal_flagged_below_min_cases(self):
        """If case_count < min_case_count (default=3), signal_flagged must be False."""
        df = self._make_ct(2, 10, 5, 500)  # a=2 < min_case_count=3
        result = self.calc.compute(df)
        assert result.iloc[0]["signal_flagged"] is False or not result.iloc[0]["signal_flagged"]

    def test_signal_flagged_for_strong_signal(self):
        """A strong signal with adequate case count should be flagged."""
        df = self._make_ct(50, 20, 10, 3000)
        result = self.calc.compute(df)
        assert result.iloc[0]["signal_flagged"]

    def test_chi2_p_value_range(self):
        """p-value must be between 0 and 1."""
        df = self._make_ct(50, 100, 30, 2000)
        result = self.calc.compute(df)
        p = result.iloc[0]["chi2_p_value"]
        assert 0.0 <= p <= 1.0
