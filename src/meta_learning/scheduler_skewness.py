"""
SchedulerSkewnessEvaluator — Measures distribution skewness across task accuracy histories.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SchedulerSkewnessEvaluator:
    """Computes statistical skewness in solve performance across curriculum task clusters."""

    @staticmethod
    def compute_skewness(recent_accuracies: List[float]) -> float:
        """Returns Fisher-Pearson coefficient of skewness for accuracy scores."""
        if len(recent_accuracies) < 3:
            return 0.0

        arr = np.array(recent_accuracies, dtype=float)
        std = np.std(arr)
        if std == 0.0:
            return 0.0

        mean = np.mean(arr)
        skewness = float(np.mean(((arr - mean) / std) ** 3))
        return skewness
