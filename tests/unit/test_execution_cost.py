"""
Unit tests for ExecutionCostEvaluator.
"""

from __future__ import annotations

import pytest

from src.meta_learning.execution_cost import ExecutionCostEvaluator


class TestExecutionCostEvaluator:
    def test_estimate_cost_cheap(self):
        cost = ExecutionCostEvaluator.estimate_cost(["rotate_90"])
        assert cost == 0.8

    def test_estimate_cost_expensive(self):
        cost = ExecutionCostEvaluator.estimate_cost(["flood_fill", "crop"])
        # flood_fill (1.5) + crop (1.0) = 2.5
        assert cost == 2.5
