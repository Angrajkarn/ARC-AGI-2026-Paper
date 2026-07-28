"""
SchedulerVelocityController — Controls adaptive step size velocity for task curriculum learning.
"""

from __future__ import annotations

from typing import List


class SchedulerVelocityController:
    """Dynamically adjusts curriculum step size based on solver accuracy trends."""

    @staticmethod
    def compute_step_size(recent_accuracies: List[float], base_step: int = 1) -> int:
        """Increases step size if recent solve accuracy is high (>0.8), decreases if low (<0.4)."""
        if not recent_accuracies:
            return base_step

        avg_acc = sum(recent_accuracies) / len(recent_accuracies)
        if avg_acc >= 0.8:
            return base_step + 2
        elif avg_acc >= 0.6:
            return base_step + 1
        elif avg_acc <= 0.3:
            return max(1, base_step - 1)

        return base_step
