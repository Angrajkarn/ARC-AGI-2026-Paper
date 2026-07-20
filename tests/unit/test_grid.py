"""Unit tests for ArcGrid core data structure."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid, COLOR_NAMES


class TestArcGrid:
    def test_from_list_basic(self):
        grid = ArcGrid.from_list([[0, 1, 2], [3, 4, 5]])
        assert grid.height == 2
        assert grid.width == 3
        assert grid.get(0, 0) == 0
        assert grid.get(1, 2) == 5

    def test_from_list_background_detection(self):
        # Most frequent color should be background
        grid = ArcGrid.from_list([[0, 0, 1], [0, 2, 0]])
        assert grid.background == 0

    def test_colors_property(self):
        grid = ArcGrid.from_list([[1, 2], [3, 1]])
        assert grid.colors == {1, 2, 3}

    def test_equality(self):
        g1 = ArcGrid.from_list([[1, 2], [3, 4]])
        g2 = ArcGrid.from_list([[1, 2], [3, 4]])
        g3 = ArcGrid.from_list([[1, 2], [3, 5]])
        assert g1 == g2
        assert g1 != g3

    def test_copy_independence(self):
        g1 = ArcGrid.from_list([[1, 2], [3, 4]])
        g2 = g1.copy()
        g2.set(0, 0, 9)
        assert g1.get(0, 0) == 1  # original unchanged

    def test_to_list_roundtrip(self):
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        grid = ArcGrid.from_list(data)
        assert grid.to_list() == data

    def test_in_bounds(self):
        grid = ArcGrid.from_list([[1, 2, 3]])
        assert grid.in_bounds(0, 0)
        assert grid.in_bounds(0, 2)
        assert not grid.in_bounds(1, 0)
        assert not grid.in_bounds(0, 3)

    def test_pixels_of_color(self):
        grid = ArcGrid.from_list([[1, 2, 1], [2, 1, 2]])
        positions = grid.pixels_of_color(1)
        assert sorted(positions) == [(0, 0), (0, 2), (1, 1)]

    def test_empty_grid(self):
        grid = ArcGrid.empty(3, 4, fill=0)
        assert grid.height == 3
        assert grid.width == 4
        assert grid.colors == {0}

    def test_hash_consistency(self):
        g1 = ArcGrid.from_list([[1, 2], [3, 4]])
        g2 = ArcGrid.from_list([[1, 2], [3, 4]])
        assert hash(g1) == hash(g2)

    def test_non_background_ratio(self):
        grid = ArcGrid.from_list([[0, 0, 1], [0, 0, 0]])
        # 1 out of 6 pixels is non-background (0)
        assert abs(grid.non_background_ratio - 1 / 6) < 0.01

    def test_color_names_complete(self):
        assert len(COLOR_NAMES) == 10
        for i in range(10):
            assert i in COLOR_NAMES
