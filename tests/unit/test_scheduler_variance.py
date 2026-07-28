"""
Unit tests for SchedulerVarianceEvaluator.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_variance import SchedulerVarianceEvaluator


class TestSchedulerVarianceEvaluator:
    def test_compute_variance_zero(self):
        # Equal scores -> 0 variance
        var = SchedulerVarianceEvaluator.compute_variance([1.0, 1.0, 1.0])
        assert var == 0.0

    def test_compute_variance_non_zero(self):
        # Varying scores
        var = SchedulerVarianceEvaluator.compute_variance([0.0, 1.0])
        assert pytest.approx(var) == 0.25
