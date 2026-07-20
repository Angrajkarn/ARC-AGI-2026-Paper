"""End-to-end integration test for the full ARC solver pipeline."""

from __future__ import annotations

import pytest

from src.api.solver_api import ARCSolver
from src.core.grid.grid import ArcGrid
from src.submission.submission_generator import SubmissionGenerator


# ---------------------------------------------------------------------------
# Sample ARC-like tasks for testing
# ---------------------------------------------------------------------------

def _make_grid(data):
    return ArcGrid.from_list(data)


# Task 1: Simple colour replacement (1 → 2)
TASK_COLOR_REPLACE = {
    "train": [
        {"input": _make_grid([[1, 0, 1], [0, 1, 0]]),
         "output": _make_grid([[2, 0, 2], [0, 2, 0]])},
        {"input": _make_grid([[1, 1, 0], [0, 0, 1]]),
         "output": _make_grid([[2, 2, 0], [0, 0, 2]])},
    ],
    "test": [
        {"input": _make_grid([[0, 1, 0], [1, 0, 1]]),
         "output": _make_grid([[0, 2, 0], [2, 0, 2]])},
    ],
}

# Task 2: Mirror horizontal
TASK_MIRROR = {
    "train": [
        {"input": _make_grid([[1, 2, 3]]),
         "output": _make_grid([[3, 2, 1]])},
        {"input": _make_grid([[1, 0], [2, 3]]),
         "output": _make_grid([[0, 1], [3, 2]])},
    ],
    "test": [
        {"input": _make_grid([[4, 5, 6]]),
         "output": _make_grid([[6, 5, 4]])},
    ],
}

# Task 3: 90-degree rotation
TASK_ROTATE = {
    "train": [
        {"input": _make_grid([[1, 2], [3, 4]]),
         "output": _make_grid([[3, 1], [4, 2]])},
        {"input": _make_grid([[5, 6], [7, 8]]),
         "output": _make_grid([[7, 5], [8, 6]])},
    ],
    "test": [
        {"input": _make_grid([[9, 1], [2, 3]]),
         "output": _make_grid([[2, 9], [3, 1]])},
    ],
}


class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def setup_solver(self):
        self.solver = ARCSolver()

    def _check_result(self, result, task_name: str):
        """Common assertions for any solver result."""
        assert result is not None, f"{task_name}: result is None"
        assert len(result.test_results) > 0, f"{task_name}: no test results"
        for tr in result.test_results:
            assert len(tr.predictions) >= 1, f"{task_name}: no predictions"
            for pred in tr.predictions:
                assert isinstance(pred, list), f"{task_name}: prediction not a list"
                assert len(pred) > 0, f"{task_name}: empty prediction grid"

    def test_color_replacement_task(self):
        result = self.solver.solve(TASK_COLOR_REPLACE, task_id="test_color_replace")
        self._check_result(result, "color_replace")

        # Check that top prediction is correct
        expected = [[0, 2, 0], [2, 0, 2]]
        top_pred = result.test_results[0].predictions[0]
        assert top_pred == expected, (
            f"Color replace: expected {expected}, got {top_pred}"
        )

    def test_mirror_task(self):
        result = self.solver.solve(TASK_MIRROR, task_id="test_mirror")
        self._check_result(result, "mirror")

        expected = [[6, 5, 4]]
        assert expected in result.test_results[0].predictions, (
            f"Mirror: expected {expected} in predictions, got {result.test_results[0].predictions}"
        )

    def test_rotate_task(self):
        result = self.solver.solve(TASK_ROTATE, task_id="test_rotate")
        self._check_result(result, "rotate")

        expected = [[2, 9], [3, 1]]
        assert expected in result.test_results[0].predictions, (
            f"Rotate: expected {expected} in predictions"
        )

    def test_two_predictions_returned(self):
        """Competition requires exactly 2 attempts per test input."""
        result = self.solver.solve(TASK_COLOR_REPLACE, task_id="test_two_preds")
        assert len(result.test_results[0].predictions) == 2

    def test_predictions_are_valid_grids(self):
        """All predictions must be non-empty 2D lists of ints 0-9."""
        result = self.solver.solve(TASK_MIRROR, task_id="test_valid_grids")
        for tr in result.test_results:
            for pred in tr.predictions:
                assert isinstance(pred, list) and len(pred) > 0
                width = len(pred[0])
                for row in pred:
                    assert len(row) == width
                    for cell in row:
                        assert isinstance(cell, int) and 0 <= cell <= 9


class TestSubmissionGenerator:
    def test_basic_generation(self):
        gen = SubmissionGenerator()
        predictions = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        gen.add_task_result("task_001", [predictions])
        data = gen.to_dict()
        assert "task_001" in data
        assert "attempt_1" in data["task_001"][0]
        assert "attempt_2" in data["task_001"][0]

    def test_save_and_reload(self, tmp_path):
        gen = SubmissionGenerator()
        gen.add_task_result("task_abc", [[[[0, 1]], [[1, 0]]]])
        output = tmp_path / "submission.json"
        gen.save(output)
        import json
        with open(output) as f:
            data = json.load(f)
        assert "task_abc" in data

    def test_fallback_for_no_predictions(self):
        gen = SubmissionGenerator()
        gen.add_task_result("task_empty", [[]])  # No predictions
        data = gen.to_dict()
        assert "attempt_1" in data["task_empty"][0]
        assert "attempt_2" in data["task_empty"][0]
