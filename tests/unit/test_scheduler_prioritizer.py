"""
Unit tests for SchedulerPrioritizer.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_prioritizer import SchedulerPrioritizer


class TestSchedulerPrioritizer:
    def test_rank_tasks(self):
        rates = {
            "task_A": 0.2,  # Hard
            "task_B": 0.9,  # Easy
            "task_C": 0.5,  # Medium
        }

        ranked = SchedulerPrioritizer.rank_tasks(rates)

        # Expected: Easy (highest success rate) first -> task_B, task_C, task_A
        assert ranked == ["task_B", "task_C", "task_A"]
