"""
SchedulerEntropyEngine — Calculates entropy-weighted probability distributions over curriculum tasks.
"""

from __future__ import annotations

import math
from typing import Dict, List


class SchedulerEntropyEngine:
    """Computes information entropy weights for active curriculum task selection."""

    @staticmethod
    def compute_entropy_weights(task_scores: Dict[str, float]) -> Dict[str, float]:
        """Returns normalized entropy-based sampling probabilities for tasks."""
        if not task_scores:
            return {}

        total = sum(task_scores.values())
        if total == 0.0:
            uniform_p = 1.0 / float(len(task_scores))
            return {k: uniform_p for k in task_scores}

        # Softmax / normalized probability calculation
        probs = {k: v / total for k, v in task_scores.items()}
        return probs
