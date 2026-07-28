"""
Unit tests for SchedulerEntropyEngine.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_entropy import SchedulerEntropyEngine


class TestSchedulerEntropyEngine:
    def test_compute_entropy_weights_empty(self):
        probs = SchedulerEntropyEngine.compute_entropy_weights({})
        assert probs == {}

    def test_compute_entropy_weights_normal(self):
        scores = {"t1": 2.0, "t2": 8.0}
        probs = SchedulerEntropyEngine.compute_entropy_weights(scores)

        assert pytest.approx(probs["t1"]) == 0.2
        assert pytest.approx(probs["t2"]) == 0.8
