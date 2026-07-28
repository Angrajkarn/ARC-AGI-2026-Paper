"""
Unit tests for SchedulerSkewnessEvaluator.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_skewness import SchedulerSkewnessEvaluator


class TestSchedulerSkewnessEvaluator:
    def test_compute_skewness_symmetric(self):
        # Symmetric scores -> 0 skewness
        skew = SchedulerSkewnessEvaluator.compute_skewness([0.2, 0.5, 0.8])
        assert pytest.approx(skew) == 0.0

    def test_compute_skewness_zero_length(self):
        skew = SchedulerSkewnessEvaluator.compute_skewness([0.5, 0.5])
        assert skew == 0.0
