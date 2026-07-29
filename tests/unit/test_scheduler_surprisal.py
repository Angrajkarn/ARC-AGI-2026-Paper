"""
Unit tests for SchedulerSurprisalEvaluator.
"""

from __future__ import annotations

import math
import pytest

from src.meta_learning.scheduler_surprisal import SchedulerSurprisalEvaluator


class TestSchedulerSurprisalEvaluator:
    def test_compute_surprisal_perfect(self):
        # Accuracy 1.0 -> -log(1.0) = 0 surprisal
        surprisal = SchedulerSurprisalEvaluator.compute_surprisal(1.0)
        assert surprisal == 0.0

    def test_compute_surprisal_partial(self):
        # Accuracy 0.5 -> -log(0.5) > 0 surprisal
        surprisal = SchedulerSurprisalEvaluator.compute_surprisal(0.5)
        assert pytest.approx(surprisal) == -math.log(0.5)
