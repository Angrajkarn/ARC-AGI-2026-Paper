"""
SchedulerInfoGainEvaluator — Measures KL divergence information gain across curriculum task updates.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class SchedulerInfoGainEvaluator:
    """Computes Kullback-Leibler (KL) divergence information gain across task solve histories."""

    @staticmethod
    def compute_information_gain(prior_probs: List[float], posterior_probs: List[float]) -> float:
        """Returns KL divergence D_KL(P_posterior || P_prior) as information gain score."""
        if len(prior_probs) != len(posterior_probs) or not prior_probs:
            return 0.0

        p = np.array(posterior_probs, dtype=float)
        q = np.array(prior_probs, dtype=float)

        # Normalize distributions safely
        p_sum = np.sum(p)
        q_sum = np.sum(q)

        if p_sum == 0.0 or q_sum == 0.0:
            return 0.0

        p = p / p_sum
        q = q / q_sum

        eps = 1e-12
        kl_div = float(np.sum(p * np.log((p + eps) / (q + eps))))
        return max(0.0, kl_div)
