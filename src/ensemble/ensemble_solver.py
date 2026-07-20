"""
Ensemble Solver — orchestrates multiple independent solvers.

Runs all configured solvers in parallel and combines results via voting.

Supported solvers:
  - symbolic   : Beam search + rule discovery
  - heuristic  : Genetic search
  - constraint : Constraint solver
  - llm_guided : LLM-assisted beam search

Combination strategy:
  - Majority vote on grid predictions.
  - Fallback to highest-scored individual result.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLProgram
from src.ranking.ranker import CandidateRanker, ScoredCandidate
from src.reasoning.rule_discovery.rule_discoverer import RuleDiscoverer
from src.search.search_engine import SearchEngine, SearchResult
from src.utils.logging_utils import get_logger
from src.verifier.verifier import Verifier

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    """Result from the ensemble solver."""

    predictions: List[List[List[int]]]   # top-2 output grids as 2-D lists
    candidates: List[ScoredCandidate]
    search_results: List[SearchResult]
    elapsed_sec: float
    found_perfect: bool
    notes: List[str] = field(default_factory=list)


class EnsembleSolver:
    """Multi-solver ensemble.

    Usage::

        solver = EnsembleSolver(solvers=["symbolic", "heuristic"])
        result = solver.solve(task, test_input)
    """

    def __init__(
        self,
        solvers: Optional[List[str]] = None,
        time_budget_sec: float = 60.0,
        max_workers: int = 2,
        max_attempts: int = 2,
        enable_llm: bool = False,
        llm_config: Optional[Dict] = None,
    ) -> None:
        self.solvers = solvers or ["symbolic", "heuristic", "constraint"]
        self.time_budget_sec = time_budget_sec
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.enable_llm = enable_llm
        self.llm_config = llm_config or {}
        self._discoverer = RuleDiscoverer(top_k=15)
        self._verifier = Verifier()
        self._ranker = CandidateRanker()
        self._executor = DSLExecutor()

    def solve(self, task: Dict, test_input: ArcGrid) -> EnsembleResult:
        """Solve one test input using ensemble of solvers.

        Args:
            task:       Full ARC task with "train" pairs (ArcGrid objects).
            test_input: Test input grid to generate prediction for.

        Returns:
            EnsembleResult with top-2 grid predictions.
        """
        t_start = time.perf_counter()
        pairs = task.get("train", [])
        notes: List[str] = []

        # Discover hypotheses (shared across all solvers)
        hypotheses = self._discoverer.discover(pairs)
        notes.append(f"Discovered {len(hypotheses)} hypotheses")

        # Run solvers
        per_budget = self.time_budget_sec / max(len(self.solvers), 1)
        search_results: List[SearchResult] = []

        if self.max_workers > 1:
            search_results = self._run_parallel(hypotheses, pairs, per_budget)
        else:
            search_results = self._run_sequential(hypotheses, pairs, per_budget)

        # Rank all found programs
        candidates = self._ranker.rank_from_search_results(
            search_results, pairs, n=self.max_attempts * 2
        )

        # Execute top candidates on test input
        predictions: List[List[List[int]]] = []
        final_candidates: List[ScoredCandidate] = []

        for cand in candidates:
            result, _ = self._executor.execute(cand.program, test_input)
            if result is not None:
                grid_list = result.to_list()
                if grid_list not in predictions:
                    predictions.append(grid_list)
                    final_candidates.append(cand)
            if len(predictions) >= self.max_attempts:
                break

        found_perfect = any(c.train_accuracy == 1.0 for c in final_candidates)

        # TTA Fallback if no 100% accurate program found yet
        if not found_perfect and hypotheses:
            from src.search.tta_wrapper import TTASearchWrapper
            engine = SearchEngine(algorithm="beam", beam_width=10, max_iterations=200)
            tta_wrapper = TTASearchWrapper(engine, num_symmetries=4)
            tta_res = tta_wrapper.search_with_tta(hypotheses, pairs, test_input, time_budget=self.time_budget_sec * 0.3)
            for tta_grid, tta_conf, tta_label in tta_res:
                grid_list = tta_grid.to_list()
                if grid_list not in predictions:
                    predictions.insert(0 if tta_conf > 0.8 else len(predictions), grid_list)
                    notes.append(f"Added TTA prediction ({tta_label}, conf={tta_conf:.2f})")
                    if len(predictions) >= self.max_attempts:
                        break

        # Fallback: use empty grid if no predictions
        while len(predictions) < self.max_attempts:
            fallback = [[0] * test_input.width for _ in range(test_input.height)]
            predictions.append(fallback)
            notes.append("Added fallback empty grid prediction")

        found_perfect = any(
            c.train_accuracy == 1.0 for c in final_candidates
        )

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Ensemble: {len(predictions)} predictions generated in {elapsed:.1f}s, "
            f"found_perfect={found_perfect}"
        )

        return EnsembleResult(
            predictions=predictions[:self.max_attempts],
            candidates=final_candidates[:self.max_attempts],
            search_results=search_results,
            elapsed_sec=elapsed,
            found_perfect=found_perfect,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Parallel / sequential runners
    # ------------------------------------------------------------------

    def _run_sequential(
        self, hypotheses: List, pairs: List, budget: float
    ) -> List[SearchResult]:
        results = []
        for solver_name in self.solvers:
            algo = self._solver_to_algo(solver_name)
            try:
                engine = SearchEngine(algorithm=algo, beam_width=10, max_iterations=300)
                result = engine.search(hypotheses, pairs, time_budget=budget)
                results.append(result)
                logger.debug(
                    f"Solver '{solver_name}': score={result.best_score:.3f}, "
                    f"perfect={result.found_perfect}"
                )
                if result.found_perfect:
                    break
            except Exception as e:
                logger.warning(f"Solver '{solver_name}' failed: {e}")
        return results

    def _run_parallel(
        self, hypotheses: List, pairs: List, budget: float
    ) -> List[SearchResult]:
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(self.max_workers, len(self.solvers))
        ) as executor:
            futures = {}
            for solver_name in self.solvers:
                algo = self._solver_to_algo(solver_name)
                engine = SearchEngine(algorithm=algo, beam_width=10, max_iterations=300)
                future = executor.submit(engine.search, hypotheses, pairs, budget)
                futures[future] = solver_name

            for future in concurrent.futures.as_completed(futures):
                solver_name = futures[future]
                try:
                    result = future.result(timeout=budget + 5)
                    results.append(result)
                    if result.found_perfect:
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    logger.warning(f"Parallel solver '{solver_name}' failed: {e}")
        return results

    @staticmethod
    def _solver_to_algo(solver_name: str) -> str:
        mapping = {
            "symbolic": "beam",
            "heuristic": "genetic",
            "constraint": "constraint",
            "llm_guided": "beam",
            "mcts": "mcts",
        }
        return mapping.get(solver_name, "beam")


__all__ = ["EnsembleSolver", "EnsembleResult"]
