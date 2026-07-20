"""
Unified Search Engine — pluggable search algorithm dispatcher.

Provides a single entry-point that selects and runs the appropriate
search algorithm based on the TaskPlan.

Supported algorithms:
  - beam      : BeamSearch
  - mcts      : Monte Carlo Tree Search
  - genetic   : GeneticSearch
  - constraint: ConstraintSolver (heuristic fallback)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.dsl.parser.dsl_parser import DSLProgram
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Unified result from any search algorithm."""

    algorithm: str
    best_program: Optional[DSLProgram]
    best_score: float
    all_candidates: List[DSLProgram] = field(default_factory=list)
    iterations: int = 0
    elapsed_sec: float = 0.0
    found_perfect: bool = False
    notes: str = ""


class SearchEngine:
    """Unified interface for all search algorithms.

    Usage::

        engine = SearchEngine(algorithm="beam", beam_width=10)
        result = engine.search(hypotheses, pairs, time_budget=60)
    """

    def __init__(
        self,
        algorithm: str = "beam",
        beam_width: int = 10,
        max_depth: int = 8,
        max_iterations: int = 500,
        parallel_workers: int = 1,
        seed: int = 42,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.algorithm = algorithm
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.parallel_workers = parallel_workers
        self.seed = seed
        self.extra_config = extra_config or {}

    def search(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 60.0,
    ) -> SearchResult:
        """Run the configured search algorithm.

        Args:
            hypotheses:  Initial hypothesis ranking from RuleDiscoverer.
            pairs:       Training pairs (dicts with ArcGrid "input"/"output").
            time_budget: Maximum wall-clock time in seconds.

        Returns:
            SearchResult with best program and metadata.
        """
        algo = self.algorithm.lower()
        logger.info(
            f"SearchEngine: starting '{algo}' search with {len(hypotheses)} hypotheses, "
            f"budget={time_budget}s"
        )

        if algo == "beam":
            return self._run_beam(hypotheses, pairs, time_budget)
        elif algo == "mcts":
            return self._run_mcts(hypotheses, pairs, time_budget)
        elif algo == "genetic":
            return self._run_genetic(hypotheses, pairs, time_budget)
        elif algo == "constraint":
            return self._run_constraint(hypotheses, pairs, time_budget)
        else:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Choose from: beam, mcts, genetic, constraint"
            )

    def search_all(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 60.0,
    ) -> List[SearchResult]:
        """Run all algorithms sequentially and return all results."""
        results = []
        per_budget = time_budget / 4
        for algo in ["beam", "mcts", "genetic", "constraint"]:
            try:
                engine = SearchEngine(
                    algorithm=algo,
                    beam_width=self.beam_width,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations,
                    seed=self.seed,
                )
                result = engine.search(hypotheses, pairs, time_budget=per_budget)
                results.append(result)
                if result.found_perfect:
                    break
            except Exception as e:
                logger.warning(f"Algorithm '{algo}' failed: {e}")
        return results

    # ------------------------------------------------------------------
    # Algorithm runners
    # ------------------------------------------------------------------

    def _run_beam(
        self, hypotheses: List[Hypothesis], pairs: List[Dict], budget: float
    ) -> SearchResult:
        from src.search.beam_search.beam_search import BeamSearch

        searcher = BeamSearch(
            beam_width=self.beam_width,
            max_depth=self.max_depth,
        )
        r = searcher.search(hypotheses, pairs, time_budget=budget, max_iterations=self.max_iterations)
        candidates = [e.program for e in r.all_candidates]
        return SearchResult(
            algorithm="beam",
            best_program=r.best_program,
            best_score=r.best_score,
            all_candidates=candidates,
            iterations=r.iterations,
            elapsed_sec=r.elapsed_sec,
            found_perfect=r.found_perfect,
        )

    def _run_mcts(
        self, hypotheses: List[Hypothesis], pairs: List[Dict], budget: float
    ) -> SearchResult:
        from src.search.mcts.mcts import MCTS

        searcher = MCTS(
            simulations=self.max_iterations,
            max_depth=self.max_depth,
            seed=self.seed,
        )
        r = searcher.search(hypotheses, pairs, time_budget=budget)
        return SearchResult(
            algorithm="mcts",
            best_program=r.best_program,
            best_score=r.best_score,
            iterations=r.iterations,
            elapsed_sec=r.elapsed_sec,
            found_perfect=r.found_perfect,
        )

    def _run_genetic(
        self, hypotheses: List[Hypothesis], pairs: List[Dict], budget: float
    ) -> SearchResult:
        from src.search.genetic.genetic_search import GeneticSearch

        pop_size = self.extra_config.get("population_size", 50)
        gens = self.extra_config.get("generations", 30)
        searcher = GeneticSearch(
            population_size=pop_size,
            generations=gens,
            seed=self.seed,
        )
        r = searcher.search(hypotheses, pairs, time_budget=budget)
        return SearchResult(
            algorithm="genetic",
            best_program=r.best_program,
            best_score=r.best_score,
            iterations=r.generation,
            elapsed_sec=r.elapsed_sec,
            found_perfect=r.found_perfect,
        )

    def _run_constraint(
        self, hypotheses: List[Hypothesis], pairs: List[Dict], budget: float
    ) -> SearchResult:
        from src.search.constraint_solver.constraint_solver import ConstraintSolver

        solver = ConstraintSolver()
        r = solver.solve(hypotheses, pairs, time_budget=budget)
        return SearchResult(
            algorithm="constraint",
            best_program=r.best_program,
            best_score=r.best_score,
            iterations=r.iterations,
            elapsed_sec=r.elapsed_sec,
            found_perfect=r.found_perfect,
        )


__all__ = ["SearchEngine", "SearchResult"]
