"""
DependencyPriors — Checks logic order execution rules across DSL instructions.
"""

from __future__ import annotations

from typing import List


class DependencyPriors:
    """Verifies that operations in program follow logical order dependencies."""

    @staticmethod
    def get_transition_score(prev_op: str, curr_op: str) -> float:
        """Assigns a compatibility score based on operation pairs."""
        # Crop operations should run early (before scaling or adding borders)
        if "crop" in prev_op and ("scale" in curr_op or "border" in curr_op):
            return 1.2
        # Avoid redundant double flips/rotations
        if prev_op == curr_op and ("rotate" in prev_op or "mirror" in prev_op):
            return 0.1
        # Recolor should come after object extraction/crop, not before
        if "recolor" in prev_op and "crop" in curr_op:
            return 0.4

        return 1.0

    def score_program(self, ops: List[str]) -> float:
        """Returns joint dependency score of the operation path sequence."""
        if len(ops) <= 1:
            return 1.0

        score = 1.0
        for i in range(len(ops) - 1):
            score *= self.get_transition_score(ops[i], ops[i + 1])

        return float(score)
