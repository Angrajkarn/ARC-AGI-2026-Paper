"""
SchedulerMomentumTracker — Tracks momentum of curriculum task performance.
"""

from __future__ import annotations

from typing import List


class SchedulerMomentumTracker:
    """Tracks solve accuracy momentum trends across task execution iterations."""

    @staticmethod
    def compute_momentum(recent_accuracies: List[float], gamma: float = 0.9) -> float:
        """Computes exponentially weighted moving average momentum score."""
        if not recent_accuracies:
            return 0.0

        momentum = 0.0
        for acc in recent_accuracies:
            momentum = gamma * momentum + (1.0 - gamma) * acc

        return momentum
