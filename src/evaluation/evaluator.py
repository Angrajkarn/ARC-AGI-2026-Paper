"""
Local Evaluator — evaluate solver performance on known ARC tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TaskEvalResult:
    task_id: str
    num_tests: int
    num_correct: int
    solve_rate: float
    elapsed_sec: float
    failure_types: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    total_tasks: int
    solved_tasks: int
    solve_rate: float
    avg_elapsed_sec: float
    task_results: List[TaskEvalResult] = field(default_factory=list)


class Evaluator:
    """Evaluate solver on a dataset of ARC tasks with known ground truth."""

    def __init__(self, solver: Any) -> None:
        self._solver = solver

    def evaluate_task(self, task_dict: Dict, task_id: str) -> TaskEvalResult:
        t = time.perf_counter()
        result = self._solver.solve(task_dict, task_id=task_id)
        elapsed = time.perf_counter() - t

        # Load test grids to check ground truth
        from src.api.solver_api import _load_task_grids
        if task_dict.get("train") and isinstance(task_dict["train"][0]["input"], list):
            task = _load_task_grids(task_dict)
        else:
            task = task_dict

        num_tests = len(task.get("test", []))
        num_correct = 0
        for i, (test_pair, test_result) in enumerate(
            zip(task.get("test", []), result.test_results)
        ):
            if test_pair.get("output") is None:
                continue
            expected = test_pair["output"].to_list()
            if expected in test_result.predictions:
                num_correct += 1

        solve_rate = num_correct / max(num_tests, 1)
        return TaskEvalResult(
            task_id=task_id,
            num_tests=num_tests,
            num_correct=num_correct,
            solve_rate=solve_rate,
            elapsed_sec=elapsed,
        )

    def evaluate_dataset(
        self, tasks: Dict[str, Dict], max_tasks: Optional[int] = None
    ) -> EvaluationSummary:
        task_ids = list(tasks.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        results = []
        for tid in task_ids:
            try:
                res = self.evaluate_task(tasks[tid], tid)
                results.append(res)
                logger.info(
                    f"[{tid}] solve_rate={res.solve_rate:.2f}, "
                    f"elapsed={res.elapsed_sec:.1f}s"
                )
            except Exception as e:
                logger.error(f"[{tid}] evaluation failed: {e}")
                results.append(TaskEvalResult(
                    task_id=tid, num_tests=0, num_correct=0,
                    solve_rate=0.0, elapsed_sec=0.0,
                    failure_types=[str(e)],
                ))

        solved = sum(1 for r in results if r.solve_rate == 1.0)
        avg_elapsed = sum(r.elapsed_sec for r in results) / max(len(results), 1)
        overall_rate = solved / max(len(results), 1)

        logger.info(
            f"Evaluation complete: {solved}/{len(results)} solved "
            f"({overall_rate:.1%}), avg={avg_elapsed:.1f}s/task"
        )

        return EvaluationSummary(
            total_tasks=len(results),
            solved_tasks=solved,
            solve_rate=overall_rate,
            avg_elapsed_sec=avg_elapsed,
            task_results=results,
        )


__all__ = ["Evaluator", "EvaluationSummary", "TaskEvalResult"]
