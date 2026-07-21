"""
Unit tests for Advanced DSL Primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.advanced_primitives import (
    dilate,
    erode,
    extract_outline,
    fill_enclosed_holes,
    pattern_repeat,
    step_ca_majority,
)
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class TestAdvancedPrimitives:
    def test_primitives_registered(self):
        assert "dilate" in PRIMITIVE_REGISTRY
        assert "erode" in PRIMITIVE_REGISTRY
        assert "extract_outline" in PRIMITIVE_REGISTRY
        assert "fill_enclosed_holes" in PRIMITIVE_REGISTRY
        assert "pattern_repeat" in PRIMITIVE_REGISTRY
        assert "step_ca_majority" in PRIMITIVE_REGISTRY

    def test_dilate_expansion(self):
        # 5x5 grid with single pixel at (2,2) of color 1
        arr = np.zeros((5, 5), dtype=int)
        arr[2, 2] = 1
        grid = ArcGrid(pixels=arr, background=0)

        dilated = dilate(grid, color=1, radius=1)
        assert dilated.get(2, 2) == 1
        assert dilated.get(1, 2) == 1
        assert dilated.get(3, 2) == 1
        assert dilated.get(2, 1) == 1
        assert dilated.get(2, 3) == 1

    def test_erode_shrinking(self):
        # 5x5 grid filled with 1, erode should shrink border
        arr = np.ones((5, 5), dtype=int)
        grid = ArcGrid(pixels=arr, background=0)

        eroded = erode(grid, color=1, radius=1)
        assert eroded.get(0, 0) == 0
        assert eroded.get(2, 2) == 1

    def test_extract_outline(self):
        # 5x5 square of color 2
        arr = np.zeros((5, 5), dtype=int)
        arr[1:4, 1:4] = 2
        grid = ArcGrid(pixels=arr, background=0)

        outline_grid = extract_outline(grid, color=2, outline_color=3)
        # Center (2,2) was interior -> should be 0 (bg)
        assert outline_grid.get(2, 2) == 0
        # Perimeter of 3x3 square should be 3
        assert outline_grid.get(1, 1) == 3
        assert outline_grid.get(1, 2) == 3

    def test_fill_enclosed_holes(self):
        # 5x5 box with hole at center (2,2)
        arr = np.zeros((5, 5), dtype=int)
        arr[1, 1:4] = 2
        arr[3, 1:4] = 2
        arr[1:4, 1] = 2
        arr[1:4, 3] = 2
        # (2,2) is background (0), completely enclosed
        grid = ArcGrid(pixels=arr, background=0)

        filled = fill_enclosed_holes(grid, fill_color=4)
        assert filled.get(2, 2) == 4
        # Outer background (0,0) remains 0
        assert filled.get(0, 0) == 0

    def test_pattern_repeat(self):
        arr = np.array([[1, 2], [3, 4]])
        grid = ArcGrid(pixels=arr, background=0)

        repeated = pattern_repeat(grid, tile_h=2, tile_w=2)
        assert repeated.height == 4
        assert repeated.width == 4
        assert repeated.get(2, 2) == 1
        assert repeated.get(3, 3) == 4

    def test_step_ca_majority(self):
        # 3x3 grid with majority 1s
        arr = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        grid = ArcGrid(pixels=arr, background=0)

        step = step_ca_majority(grid)
        assert step.get(1, 1) == 1
