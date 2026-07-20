"""
Submission Generator — produce competition-ready submission.json.

ARC Prize submission format (per test input, 2 attempts):
{
  "<task_id>": [
    {
      "attempt_1": [[0,1,2], ...],
      "attempt_2": [[0,1,2], ...]
    },
    ...  (one dict per test input in the task)
  ],
  ...
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SubmissionGenerator:
    """Generate submission.json for the ARC Prize competition.

    Usage::

        gen = SubmissionGenerator()
        gen.add_task_result("007bbfb7", predictions=[[grid1, grid2], [grid3, grid4]])
        gen.save("submission.json")
    """

    def __init__(self, validate: bool = True) -> None:
        self.validate = validate
        self._submission: Dict[str, Any] = {}

    def add_task_result(
        self,
        task_id: str,
        predictions: List[List[List[List[int]]]],
    ) -> None:
        """Add predictions for one task.

        Args:
            task_id:     Task identifier (filename stem).
            predictions: For each test input: a list of up to 2 candidate grids.
                         predictions[i] = [attempt_1_grid, attempt_2_grid]
        """
        task_entry = []
        for test_idx, attempts in enumerate(predictions):
            entry: Dict[str, Any] = {}
            for attempt_num, grid in enumerate(attempts[:2], start=1):
                entry[f"attempt_{attempt_num}"] = grid
            # If fewer than 2 attempts, duplicate the first
            if len(attempts) == 1:
                entry["attempt_2"] = attempts[0]
            elif len(attempts) == 0:
                # Fallback: 1x1 grid of zeros
                entry["attempt_1"] = [[0]]
                entry["attempt_2"] = [[0]]
            task_entry.append(entry)

        self._submission[task_id] = task_entry

        if self.validate:
            errors = self._validate_task_entry(task_id, task_entry)
            if errors:
                logger.warning(f"Submission validation warnings for {task_id}: {errors}")

    def add_from_solver_result(self, solver_result: Any) -> None:
        """Convenience: add from a SolverResult object.

        Args:
            solver_result: SolverResult from ARCSolver.solve()
        """
        # Group predictions by test input
        predictions = []
        for test_result in solver_result.test_results:
            predictions.append(test_result.predictions)

        self.add_task_result(solver_result.task_id, predictions)

    def save(self, output_path: str | Path, indent: int = 2) -> None:
        """Write submission.json to disk.

        Args:
            output_path: Output file path.
            indent:      JSON indentation (0 for compact).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.validate:
            all_errors = self._validate_submission()
            if all_errors:
                logger.warning(f"Submission has {len(all_errors)} validation issues")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._submission,
                f,
                indent=indent if indent > 0 else None,
            )

        n_tasks = len(self._submission)
        n_tests = sum(len(v) for v in self._submission.values())
        logger.info(
            f"Submission saved: {path} "
            f"({n_tasks} tasks, {n_tests} test inputs)"
        )

    def to_dict(self) -> Dict:
        return dict(self._submission)

    def summary(self) -> str:
        n_tasks = len(self._submission)
        n_tests = sum(len(v) for v in self._submission.values())
        return f"SubmissionGenerator: {n_tasks} tasks, {n_tests} total test inputs"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_task_entry(self, task_id: str, entry: List[Dict]) -> List[str]:
        errors = []
        for i, test_dict in enumerate(entry):
            for key in ("attempt_1", "attempt_2"):
                if key not in test_dict:
                    errors.append(f"[{task_id}][{i}] missing {key}")
                    continue
                grid = test_dict[key]
                if not isinstance(grid, list) or not grid:
                    errors.append(f"[{task_id}][{i}][{key}] empty or non-list grid")
                    continue
                width = len(grid[0]) if grid else 0
                for row_idx, row in enumerate(grid):
                    if not isinstance(row, list) or len(row) != width:
                        errors.append(
                            f"[{task_id}][{i}][{key}][{row_idx}] jagged row"
                        )
                    for cell in row:
                        if not isinstance(cell, int) or not (0 <= cell <= 9):
                            errors.append(
                                f"[{task_id}][{i}][{key}] invalid cell value {cell!r}"
                            )
        return errors

    def _validate_submission(self) -> List[str]:
        errors = []
        for task_id, entry in self._submission.items():
            errors.extend(self._validate_task_entry(task_id, entry))
        return errors


__all__ = ["SubmissionGenerator"]
