"""
ExecutionCostEvaluator — Estimates AST program statement execution complexity costs.
"""

from __future__ import annotations

from typing import List


class ExecutionCostEvaluator:
    """Estimates computational penalty scores based on AST program depth and statement counts."""

    @staticmethod
    def estimate_cost(ops: List[str]) -> float:
        """Assigns higher cost values to longer or nested primitive loops."""
        cost = 0.0
        for op in ops:
            if "fill" in op or "recolor" in op:
                cost += 1.5  # Complex pixel iteration op
            elif "crop" in op or "mirror" in op:
                cost += 1.0  # Basic dimension transform op
            else:
                cost += 0.8  # Cheap element property query op
        return cost
