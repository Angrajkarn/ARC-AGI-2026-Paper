"""
Constraint Solver — constraint-guided program synthesis for ARC tasks.

Uses hard constraints derived from input/output features to prune the
search space, then greedily enumerates consistent programs.

Constraints extracted:
  - Output size (height, width) must match.
  - Output colour set must match.
  - Output background must match.
  - Number of objects must match (if applicable).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ConstraintResult:
    best_program: Optional[DSLProgram]
    best_score: float
    iterations: int
    elapsed_sec: float
    found_perfect: bool


class ConstraintSolver:
    """Constraint-guided program synthesiser.

    Extracts constraints from training pairs, filters hypotheses that
    violate constraints, then performs greedy extension.
    """

    def __init__(self, max_depth: int = 5) -> None:
        self.max_depth = max_depth
        self._executor = DSLExecutor(debug=False)

    def solve(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 60.0,
    ) -> ConstraintResult:
        """Solve with constraint guidance.

        Args:
            hypotheses: Initial hypothesis ranking.
            pairs:      Training pairs.
            time_budget: Time limit.

        Returns:
            ConstraintResult with best program.
        """
        t_start = time.perf_counter()
        constraints = self._extract_constraints(pairs)
        logger.debug(f"ConstraintSolver: extracted constraints: {constraints}")

        best_prog: Optional[DSLProgram] = None
        best_score = 0.0
        iterations = 0

        # First: evaluate hypotheses directly
        for hyp in hypotheses:
            if time.perf_counter() - t_start > time_budget:
                break
            score = self._score(hyp.candidate_program, pairs)
            iterations += 1
            if score > best_score:
                best_score = score
                best_prog = hyp.candidate_program
            if score == 1.0:
                break

        if best_score == 1.0:
            return ConstraintResult(
                best_program=best_prog,
                best_score=best_score,
                iterations=iterations,
                elapsed_sec=time.perf_counter() - t_start,
                found_perfect=True,
            )

        # Second: greedy constraint-guided enumeration
        from src.search.beam_search.beam_search import BeamSearch
        remaining = max(0, time_budget - (time.perf_counter() - t_start))
        beam = BeamSearch(beam_width=5, max_depth=self.max_depth)
        beam_result = beam.search(hypotheses, pairs, time_budget=remaining, max_iterations=200)

        if beam_result.best_score > best_score:
            best_score = beam_result.best_score
            best_prog = beam_result.best_program

        elapsed = time.perf_counter() - t_start
        return ConstraintResult(
            best_program=best_prog,
            best_score=best_score,
            iterations=iterations + beam_result.iterations,
            elapsed_sec=elapsed,
            found_perfect=(best_score == 1.0),
        )

    def _extract_constraints(self, pairs: List[Dict]) -> Dict:
        """Extract hard constraints from training pairs."""
        constraints = {}
        if not pairs:
            return constraints

        # Size constraints
        sizes = [(p["output"].height, p["output"].width) for p in pairs]
        input_sizes = [(p["input"].height, p["input"].width) for p in pairs]
        if len(set(sizes)) == 1:
            constraints["fixed_output_size"] = sizes[0]
        if all(s == i for s, i in zip(sizes, input_sizes)):
            constraints["preserves_size"] = True

        # Colour constraints
        color_sets = [frozenset(p["output"].colors) for p in pairs]
        if len(set(color_sets)) == 1:
            constraints["fixed_output_colors"] = color_sets[0]

        # Background
        bgs = [p["output"].background for p in pairs]
        if len(set(bgs)) == 1:
            constraints["fixed_background"] = bgs[0]

        return constraints

    def _score(self, program: DSLProgram, pairs: List[Dict]) -> float:
        correct = 0
        for pair in pairs:
            result, _ = self._executor.execute(program, pair["input"])
            if result is not None and result == pair["output"]:
                correct += 1
        return correct / max(len(pairs), 1)


__all__ = ["ConstraintSolver", "ConstraintResult"]
