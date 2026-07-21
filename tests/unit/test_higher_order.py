"""
Unit tests for higher-order AST primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.higher_order import conditional_apply, map_objects, reduce_lines
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class TestHigherOrderPrimitives:
    def test_higher_order_registered(self):
        assert "map_objects" in PRIMITIVE_REGISTRY
        assert "conditional_apply" in PRIMITIVE_REGISTRY
        assert "reduce_lines" in PRIMITIVE_REGISTRY

    def test_map_objects_rotate(self):
        arr = np.zeros((6, 6), dtype=int)
        # Add 2x2 square at top left
        arr[1:3, 1:3] = 1
        grid = ArcGrid(pixels=arr, background=0)

        mapped = map_objects(grid, primitive_name="rotate_90")
        assert mapped.height == 6
        assert mapped.width == 6

    def test_conditional_apply(self):
        # Symmetric grid
        arr = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])
        grid = ArcGrid(pixels=arr, background=0)

        res = conditional_apply(grid, condition="is_symmetric", true_primitive="rotate_90", false_primitive="mirror_horizontal")
        assert res.height == 3
        assert res.width == 3

    def test_reduce_lines(self):
        arr = np.array([[0, 1, 0], [0, 0, 2]])
        grid = ArcGrid(pixels=arr, background=0)

        reduced_rows = reduce_lines(grid, axis=0)
        assert reduced_rows.height == 1
        assert reduced_rows.width == 3
        assert reduced_rows.get(0, 1) == 1
        assert reduced_rows.get(0, 2) == 2
