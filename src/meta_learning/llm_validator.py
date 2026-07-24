"""
LLMValidator — Validates candidate python programs against training grid examples.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

from src.core.grid.grid import ArcGrid


class LLMValidator:
    """Simulates multi-model consensus verification on synthesized solution snippets."""

    @staticmethod
    def validate_candidate(
        candidate_fn: Callable[[ArcGrid], ArcGrid],
        examples: List[Tuple[ArcGrid, ArcGrid]]
    ) -> float:
        """Runs candidate python function against demonstration examples and returns accuracy score."""
        if not examples:
            return 0.0

        passed = 0
        for inp, out in examples:
            try:
                res = candidate_fn(inp)
                if (res.pixels == out.pixels).all():
                    passed += 1
            except Exception:
                pass

        return float(passed) / float(len(examples))
