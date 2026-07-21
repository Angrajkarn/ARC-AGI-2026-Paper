"""
Unit tests for ARCMagics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.ui.jupyter_magic import ARCMagics, load_ipython_extension


class TestJupyterMagic:
    def test_visualize_grid_magic(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        magics = ARCMagics()

        res = magics.visualize_grid(grid)
        assert "ArcGrid(2x2" in res

    def test_solve_task_cell(self):
        magics = ARCMagics()
        res = magics.solve_task_cell("007bbfb7.json")
        assert "Solved task" in res

    def test_load_extension(self):
        load_ipython_extension(None)
