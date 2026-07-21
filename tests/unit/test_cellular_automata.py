"""
Unit tests for cellular automata primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.cellular_automata import (
    ca_game_of_life,
    ca_sand_fall,
    ca_water_fill,
)


class TestCellularAutomata:
    def test_ca_sand_fall(self):
        grid = ArcGrid(pixels=np.array([[1, 0], [0, 0]]), background=0)
        res = ca_sand_fall(grid, sand_color=1)
        assert res.get(1, 0) == 1
        assert res.get(0, 0) == 0

    def test_ca_water_fill(self):
        grid = ArcGrid(pixels=np.array([[8, 0], [0, 0]]), background=0)
        res = ca_water_fill(grid, water_color=8, fill_color=1)
        assert res.get(0, 1) == 1
        assert res.get(1, 0) == 1

    def test_ca_game_of_life(self):
        grid = ArcGrid(pixels=np.array([[1, 1], [1, 0]]), background=0)
        res = ca_game_of_life(grid, live_color=1)
        assert res.get(1, 1) == 1
