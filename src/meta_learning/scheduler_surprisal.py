"""
SchedulerSurprisalEvaluator — Measures Bayesian surprisal (-log P(x)) across task solve accuracy trajectories.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class SchedulerSurprisalEvaluator:
    """Computes negative log-likelihood Bayesian surprisal for curriculum task performance."""

    @staticmethod
    def compute_surprisal(accuracy: float, epsilon: float = 1e-6) -> float:
        """Returns Bayesian surprisal score -log(accuracy) in [0.0, inf)."""
        safe_acc = float(np.clip(accuracy, epsilon, 1.0))
        return float(-math.log(safe_acc))
