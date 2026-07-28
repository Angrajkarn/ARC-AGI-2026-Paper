"""
Unit tests for SchedulerMomentumTracker.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_momentum import SchedulerMomentumTracker


class TestSchedulerMomentumTracker:
    def test_compute_momentum_zero(self):
        mom = SchedulerMomentumTracker.compute_momentum([])
        assert mom == 0.0

    def test_compute_momentum_trend(self):
        accuracies = [1.0, 1.0, 1.0]
        mom = SchedulerMomentumTracker.compute_momentum(accuracies, gamma=0.5)
        # acc 1: 0.5 * 0 + 0.5 * 1 = 0.5
        # acc 2: 0.5 * 0.5 + 0.5 * 1 = 0.75
        # acc 3: 0.5 * 0.75 + 0.5 * 1 = 0.875
        assert pytest.approx(mom) == 0.875
