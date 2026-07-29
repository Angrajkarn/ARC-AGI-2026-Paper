"""
Unit tests for SchedulerInfoGainEvaluator.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_info_gain import SchedulerInfoGainEvaluator


class TestSchedulerInfoGainEvaluator:
    def test_compute_information_gain_same(self):
        # Identical distributions -> 0 KL divergence
        gain = SchedulerInfoGainEvaluator.compute_information_gain([0.5, 0.5], [0.5, 0.5])
        assert pytest.approx(gain, abs=1e-5) == 0.0

    def test_compute_information_gain_different(self):
        # Different distributions -> positive KL divergence
        gain = SchedulerInfoGainEvaluator.compute_information_gain([0.5, 0.5], [0.9, 0.1])
        assert gain > 0.0
