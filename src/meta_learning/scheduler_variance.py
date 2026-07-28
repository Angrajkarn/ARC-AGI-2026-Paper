"""
SchedulerVarianceEvaluator — Calculates variance metrics across task solve performance histories.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SchedulerVarianceEvaluator:
    """Computes statistical variance in task solve performance."""

    @staticmethod
    def compute_variance(recent_accuracies: List[float]) -> float:
        """Returns population variance of recent solve accuracy scores."""
        if len(recent_accuracies) < 2:
            return 0.0

        return float(np.var(recent_accuracies))
