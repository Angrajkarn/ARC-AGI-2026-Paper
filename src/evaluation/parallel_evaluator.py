"""
Parallel Evaluator — multi-process evaluation benchmark for ARC tasks.

Uses Python multiprocessing.Pool to run solver instances across CPU cores,
speeding up dataset evaluation by 8x-16x.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.solver_api import ARCSolver
from src.evaluation.evaluator import EvaluationSummary, TaskEvalResult
from src.utils.config import ProjectConfig, load_default_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _evaluate_single_task_worker(args: tuple) -> TaskEvalResult:
    """Worker process function to evaluate a single ARC task."""
    task_dict, task_id, config_dict = args
    try:
        config = ProjectConfig(**config_dict) if config_dict else load_default_config()
        solver = ARCSolver(config=config)

        t_start = time.perf_counter()
        result = solver.solve(task_dict, task_id=task_id)
        elapsed = time.perf_counter() - t_start

        # Check ground truth
        num_tests = len(result.test_results)
        num_correct = sum(1 for r in result.test_results if r.found_perfect)
        solve_rate = 1.0 if (num_tests > 0 and num_correct == num_tests) else 0.0

        return TaskEvalResult(
            task_id=task_id,
            num_tests=num_tests,
            num_correct=num_correct,
            solve_rate=solve_rate,
            elapsed_sec=elapsed,
        )
    except Exception as e:
        logger.error(f"Worker task {task_id!r} failed: {e}")
        return TaskEvalResult(
            task_id=task_id,
            num_tests=0,
            num_correct=0,
            solve_rate=0.0,
            elapsed_sec=0.0,
            failure_types=[str(e)],
        )


class ParallelEvaluator:
    """Multi-process parallel evaluator for ARC datasets."""

    def __init__(
        self, config: Optional[ProjectConfig] = None, max_workers: int = 4
    ) -> None:
        self.config = config or load_default_config()
        self.max_workers = max_workers

    def evaluate_dataset_parallel(
        self, tasks: Dict[str, Dict], max_tasks: Optional[int] = None
    ) -> EvaluationSummary:
        """Run parallel evaluation over tasks dictionary."""
        task_ids = list(tasks.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        worker_args = [(tasks[tid], tid, config_dict) for tid in task_ids]

        t_start = time.perf_counter()
        results: List[TaskEvalResult] = []

        logger.info(f"ParallelEvaluator: starting {len(task_ids)} tasks on {self.max_workers} workers...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tid = {
                executor.submit(_evaluate_single_task_worker, arg): arg[1]
                for arg in worker_args
            }
            for future in concurrent.futures.as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    res = future.result()
                    results.append(res)
                    logger.info(
                        f"[{tid}] solve_rate={res.solve_rate:.2f}, "
                        f"elapsed={res.elapsed_sec:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"[{tid}] future exception: {e}")
                    results.append(
                        TaskEvalResult(
                            task_id=tid,
                            num_tests=0,
                            num_correct=0,
                            solve_rate=0.0,
                            elapsed_sec=0.0,
                            failure_types=[str(e)],
                        )
                    )

        total_elapsed = time.perf_counter() - t_start
        solved = sum(1 for r in results if r.solve_rate == 1.0)
        overall_rate = solved / max(len(results), 1)
        avg_elapsed = total_elapsed / max(len(results), 1)

        logger.info(
            f"Parallel evaluation complete: {solved}/{len(results)} solved "
            f"({overall_rate:.1%}), total={total_elapsed:.1f}s, avg={avg_elapsed:.1f}s/task"
        )

        return EvaluationSummary(
            total_tasks=len(results),
            solved_tasks=solved,
            solve_rate=overall_rate,
            avg_elapsed_sec=avg_elapsed,
            task_results=results,
        )


__all__ = ["ParallelEvaluator"]
