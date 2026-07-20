"""
Main Solver API — clean Python interface to the ARC solver pipeline.

Usage::

    from src.api.solver_api import ARCSolver
    from src.utils.io import load_task

    solver = ARCSolver()
    task = load_task("data/datasets/training/007bbfb7.json")
    result = solver.solve(task)
    print(result.predictions)   # List of 2-D grid predictions per test input
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.ensemble.ensemble_solver import EnsembleSolver
from src.memory.memory_store import MemoryStore
from src.reasoning.planner.task_planner import TaskPlan, TaskPlanner
from src.reflection.reflector import Reflector
from src.search.search_engine import SearchEngine
from src.utils.config import ProjectConfig, load_default_config
from src.utils.io import ARCTask, Grid
from src.utils.logging_utils import get_logger
from src.verifier.verifier import Verifier

logger = get_logger(__name__)


@dataclass
class TestInputResult:
    """Prediction for a single test input."""

    test_index: int
    predictions: List[Grid]          # Up to 2 candidate grids
    confidence: float
    elapsed_sec: float
    found_perfect: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class SolverResult:
    """Complete result for a full ARC task."""

    task_id: str
    test_results: List[TestInputResult]
    total_elapsed_sec: float
    all_correct: bool = False         # True if all test predictions verified (gold label known)
    notes: List[str] = field(default_factory=list)

    @property
    def predictions(self) -> List[List[Grid]]:
        """List of prediction-lists per test input."""
        return [r.predictions for r in self.test_results]


def _load_task_grids(task_dict: Dict) -> Dict:
    """Convert raw ARC task dicts (2-D list grids) to ArcGrid objects."""
    def _to_grid(raw: List[List[int]]) -> ArcGrid:
        return ArcGrid.from_list(raw)

    converted_train = []
    for pair in task_dict.get("train", []):
        converted_train.append({
            "input": _to_grid(pair["input"]),
            "output": _to_grid(pair["output"]),
        })

    converted_test = []
    for pair in task_dict.get("test", []):
        converted_test.append({
            "input": _to_grid(pair["input"]),
            "output": _to_grid(pair["output"]) if "output" in pair else None,
        })

    return {"train": converted_train, "test": converted_test}


class ARCSolver:
    """Main ARC solver — entry point for the full pipeline.

    Usage::

        solver = ARCSolver()
        result = solver.solve(task_dict, task_id="my_task")
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or load_default_config()
        self._planner = TaskPlanner(config=self.config)
        self._verifier = Verifier()
        self._reflector = Reflector()
        self._memory = MemoryStore(
            backend=self.config.memory.backend,
        )
        self._build_ensemble()

    def _build_ensemble(self) -> None:
        cfg = self.config.solver
        self._ensemble = EnsembleSolver(
            solvers=cfg.solvers,
            time_budget_sec=cfg.max_time_per_task,
            max_attempts=cfg.max_attempts,
            enable_llm=cfg.enable_llm,
            llm_config=self.config.llm.model_dump() if hasattr(self.config.llm, "model_dump") else {},
        )

    def solve(self, task_dict: Dict, task_id: str = "unknown") -> SolverResult:
        """Solve a complete ARC task.

        Args:
            task_dict: Raw ARC task dict (with 2-D list grids or ArcGrid objects).
            task_id:   Optional human-readable task identifier.

        Returns:
            SolverResult with predictions for all test inputs.
        """
        t_start = time.perf_counter()
        logger.info(f"Solving task {task_id!r}...")

        # Normalise grids
        if task_dict.get("train") and isinstance(task_dict["train"][0]["input"], list):
            task = _load_task_grids(task_dict)
        else:
            task = task_dict  # already ArcGrid objects

        # Plan
        plan = self._planner.plan(task)
        logger.info(
            f"Plan: difficulty={plan.difficulty}, algo={plan.search_algorithm}, "
            f"hypotheses={len(plan.hypotheses)}"
        )

        # Solve each test input
        test_results: List[TestInputResult] = []
        for i, test_pair in enumerate(task.get("test", [])):
            test_input: ArcGrid = test_pair["input"]
            t_test = time.perf_counter()

            ens_result = self._ensemble.solve(task, test_input)

            # Reflection loop if not perfect
            if not ens_result.found_perfect and plan.hypotheses:
                ens_result = self._reflect_and_retry(task, test_input, ens_result, plan)

            elapsed = time.perf_counter() - t_test
            predictions = ens_result.predictions

            # Check against ground truth if available
            is_correct = False
            if test_pair.get("output") is not None:
                expected = test_pair["output"].to_list()
                is_correct = expected in predictions

            test_results.append(TestInputResult(
                test_index=i,
                predictions=predictions,
                confidence=ens_result.candidates[0].score if ens_result.candidates else 0.0,
                elapsed_sec=elapsed,
                found_perfect=ens_result.found_perfect,
                notes=ens_result.notes,
            ))
            logger.info(
                f"  Test [{i}]: {len(predictions)} predictions, "
                f"confidence={test_results[-1].confidence:.3f}, "
                f"correct={is_correct}"
            )

        total_elapsed = time.perf_counter() - t_start
        result = SolverResult(
            task_id=task_id,
            test_results=test_results,
            total_elapsed_sec=total_elapsed,
        )
        logger.info(
            f"Task {task_id!r} done in {total_elapsed:.1f}s"
        )
        return result

    def _reflect_and_retry(self, task, test_input, ens_result, plan) -> Any:
        """Run one reflection-guided retry."""
        pairs = task.get("train", [])
        if not ens_result.candidates:
            return ens_result

        best_cand = ens_result.candidates[0]
        verification = self._verifier.verify(best_cand.program, pairs)
        report = self._reflector.reflect(best_cand.program, pairs, verification)

        if not report.modified_programs:
            return ens_result

        # Quick re-run on modified programs
        from src.dsl.executor.executor import DSLExecutor
        executor = DSLExecutor()
        new_predictions = list(ens_result.predictions)

        for mod_prog in report.modified_programs[:3]:
            ver = self._verifier.verify(mod_prog, pairs)
            if ver.score > best_cand.train_accuracy:
                result, _ = executor.execute(mod_prog, test_input)
                if result is not None:
                    grid = result.to_list()
                    if grid not in new_predictions:
                        new_predictions.insert(0, grid)
                        logger.debug(f"Reflection improved prediction (score {ver.score:.2f})")
                        break

        ens_result.predictions = new_predictions[:2]
        return ens_result


__all__ = ["ARCSolver", "SolverResult", "TestInputResult"]
