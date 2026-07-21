"""
Unit tests for ParallelEvaluator.
"""

from __future__ import annotations

import pytest

from src.evaluator.parallel_evaluator import ParallelEvaluator


class TestParallelEvaluator:
    def test_evaluate_batch(self):
        tasks = {
            "task1": {"train": [{"input": [[1]], "output": [[1]]}]},
            "task2": {"train": [{"input": [[2]], "output": [[2]]}]},
        }

        evaluator = ParallelEvaluator(max_workers=2)
        results = evaluator.evaluate_batch(tasks)

        assert len(results) == 2
        assert all(r.solved for r in results)
