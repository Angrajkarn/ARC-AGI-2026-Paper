"""
SchedulerRateDistortionEvaluator — Computes Rate-Distortion R(D) code length bounds for task curriculum sets.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class SchedulerRateDistortionEvaluator:
    """Estimates theoretical rate-distortion R(D) code length lower bounds."""

    @staticmethod
    def compute_rate_distortion_bound(distortion: float, max_rate: float = 10.0) -> float:
        """Returns rate-distortion lower bound R(D) = max(0, -log2(distortion + eps))."""
        safe_d = float(np.clip(distortion, 1e-6, 1.0))
        rate = float(-math.log2(safe_d))
        return float(min(rate, max_rate))
