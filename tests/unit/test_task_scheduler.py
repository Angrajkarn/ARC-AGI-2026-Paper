"""
Unit tests for TaskCurriculumScheduler.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from src.meta_learning.task_scheduler import TaskCurriculumScheduler


class TestTaskCurriculumScheduler:
    def test_schedule_curriculum(self):
        # Create temp dir with dummy tasks
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            task_simple = {
                "train": [
                    {"input": [[1]], "output": [[1]]}  # size factor = 2, colors = 1 -> score 2
                ]
            }

            task_complex = {
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}  # size factor = 8, colors = 4 -> score 32
                ]
            }

            with open(path / "simple.json", "w") as f:
                json.dump(task_simple, f)
            with open(path / "complex.json", "w") as f:
                json.dump(task_complex, f)

            scheduler = TaskCurriculumScheduler()
            ordered = scheduler.schedule_curriculum(path)

            assert len(ordered) == 2
            # simple.json should come first because score 2 < 32
            assert ordered[0].name == "simple.json"
            assert ordered[1].name == "complex.json"
