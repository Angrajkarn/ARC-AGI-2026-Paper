"""
Unit tests for SchedulerRateDistortionEvaluator.
"""

from __future__ import annotations

import math
import pytest

from src.meta_learning.scheduler_rate_distortion import SchedulerRateDistortionEvaluator


class TestSchedulerRateDistortionEvaluator:
    def test_compute_rate_distortion_bound_zero(self):
        # Distortion 1.0 -> -log2(1.0) = 0 rate
        rate = SchedulerRateDistortionEvaluator.compute_rate_distortion_bound(1.0)
        assert rate == 0.0

    def test_compute_rate_distortion_bound_half(self):
        # Distortion 0.5 -> -log2(0.5) = 1 bit rate
        rate = SchedulerRateDistortionEvaluator.compute_rate_distortion_bound(0.5)
        assert pytest.approx(rate) == 1.0
