"""
SchedulerPrioritizer — Ranks task curriculum indices based on performance score weightings.
"""

from __future__ import annotations

from typing import Dict, List


class SchedulerPrioritizer:
    """Prioritizes tasks by historical solve failure rates to order simple-to-complex curves."""

    @staticmethod
    def rank_tasks(task_success_rates: Dict[str, float]) -> List[str]:
        """Returns sorted task names ascending by success rate (hardest tasks last)."""
        # Sort tasks: highest success rate (easiest) first
        sorted_pairs = sorted(task_success_rates.items(), key=lambda item: item[1], reverse=True)
        return [task_name for task_name, _ in sorted_pairs]
