"""
Unit tests for TaskCurriculumEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.genetic.curriculum import TaskCurriculumEngine


class TestTaskCurriculum:
    def test_sort_pairs(self):
        # Easiest: uniform color 2x2 grid
        g_easy = ArcGrid(pixels=np.array([[1, 1], [1, 1]]), background=0)
        # Harder: multi-colored 3x3 grid
        g_hard = ArcGrid(pixels=np.array([[1, 2, 3], [4, 5, 0], [1, 2, 3]]), background=0)

        pairs = [
            {"input": g_hard, "output": g_hard},
            {"input": g_easy, "output": g_easy},
        ]

        engine = TaskCurriculumEngine()
        sorted_pairs = engine.sort_pairs(pairs)

        # Easy grid should be positioned first
        assert sorted_pairs[0]["input"] == g_easy
        assert sorted_pairs[1]["input"] == g_hard
