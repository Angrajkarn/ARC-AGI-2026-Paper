"""
ParallelEvaluator — Multi-core process pool task evaluation engine.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.core.grid.grid import ArcGrid


@dataclass
class EvalTaskResult:
    task_id: str
    solved: bool
    runtime: float


def _evaluate_single_task(args: Tuple[str, Dict[str, Any]]) -> EvalTaskResult:
    """Worker function for single task evaluation."""
    task_id, task_data = args
    start_t = time.time()

    # Simple dummy evaluation verification
    train_pairs = task_data.get("train", [])
    solved = len(train_pairs) > 0
    runtime = time.time() - start_t

    return EvalTaskResult(task_id=task_id, solved=solved, runtime=runtime)


class ParallelEvaluator:
    """Evaluates batches of ARC tasks in parallel using multiprocessing pool."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def evaluate_batch(self, tasks: Dict[str, Dict[str, Any]]) -> List[EvalTaskResult]:
        """Evaluates dictionary of tasks in parallel."""
        task_items = list(tasks.items())
        results: List[EvalTaskResult] = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for res in executor.map(_evaluate_single_task, task_items):
                results.append(res)

        return results
