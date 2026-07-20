"""Unit tests for competitive extensions: ShapePredictor, Higher-Order Combinators, TTA, and new primitives."""

from __future__ import annotations

import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.combinators import (
    complete_symmetry_h,
    complete_symmetry_v,
    connect_points_of_color,
    filter_objects_by_color,
    filter_objects_by_size,
    filter_objects_by_shape,
    overlay,
)
from src.dsl.primitives.transforms import (
    center_content,
    invert_colors,
    isolate_largest,
    isolate_smallest,
    recolor_non_background,
)
from src.reasoning.planner.shape_predictor import ShapePredictor
from src.search.tta_wrapper import invert_grid, transform_grid


def make_grid(data, bg=0):
    return ArcGrid.from_list(data, background=bg)


class TestShapePredictor:
    def test_fixed_size_prediction(self):
        pairs = [
            {"input": make_grid([[1, 2], [3, 4]]), "output": make_grid([[5, 5, 5], [5, 5, 5], [5, 5, 5]])},
            {"input": make_grid([[1, 1, 1], [2, 2, 2]]), "output": make_grid([[5, 5, 5], [5, 5, 5], [5, 5, 5]])},
        ]
        predictor = ShapePredictor()
        test_in = make_grid([[9, 9]])
        preds = predictor.predict(pairs, test_in)
        assert preds[0].height == 3
        assert preds[0].width == 3
        assert preds[0].rule_type == "fixed"

    def test_identity_size_prediction(self):
        pairs = [
            {"input": make_grid([[1, 2], [3, 4]]), "output": make_grid([[1, 2], [3, 4]])},
            {"input": make_grid([[1, 1, 1]]), "output": make_grid([[1, 1, 1]])},
        ]
        predictor = ShapePredictor()
        test_in = make_grid([[9, 8, 7], [6, 5, 4]])
        preds = predictor.predict(pairs, test_in)
        assert preds[0].height == 2
        assert preds[0].width == 3
        assert preds[0].rule_type == "identity"

    def test_scale_prediction(self):
        pairs = [
            {"input": make_grid([[1, 2]]), "output": make_grid([[1, 1, 2, 2], [1, 1, 2, 2]])},
            {"input": make_grid([[1, 2], [3, 4]]), "output": make_grid([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])},
        ]
        predictor = ShapePredictor()
        test_in = make_grid([[3, 3], [3, 3]])
        preds = predictor.predict(pairs, test_in)
        assert preds[0].height == 4
        assert preds[0].width == 4


class TestCombinators:
    def test_filter_objects_by_color(self):
        grid = make_grid([[1, 1, 0, 2], [1, 1, 0, 2]])
        res = filter_objects_by_color(grid, color=1)
        assert res.get(0, 0) == 1
        assert res.get(0, 3) == 0  # color 2 removed

    def test_filter_objects_by_size(self):
        grid = make_grid([[1, 1, 0, 2], [1, 1, 0, 0]])  # 1s area=4, 2s area=1
        res = filter_objects_by_size(grid, mode="largest")
        assert res.get(0, 0) == 1
        assert res.get(0, 3) == 0

    def test_overlay(self):
        a = make_grid([[1, 1], [1, 1]])
        b = make_grid([[0, 2], [0, 0]])
        res = overlay(a, b, transparent_color=0)
        assert res.to_list() == [[1, 2], [1, 1]]

    def test_connect_points_of_color(self):
        grid = make_grid([[1, 0, 0, 1]])
        res = connect_points_of_color(grid, color=1)
        assert res.to_list() == [[1, 1, 1, 1]]

    def test_complete_symmetry_h(self):
        grid = make_grid([[1, 0, 0, 0]])
        res = complete_symmetry_h(grid)
        assert res.get(0, 3) == 1


class TestNewPrimitives:
    def test_invert_colors(self):
        grid = make_grid([[0, 1], [2, 9]])
        res = invert_colors(grid)
        assert res.to_list() == [[9, 8], [7, 0]]

    def test_recolor_non_background(self):
        grid = make_grid([[0, 2], [3, 0]], bg=0)
        res = recolor_non_background(grid, new_color=5)
        assert res.to_list() == [[0, 5], [5, 0]]

    def test_isolate_largest(self):
        grid = make_grid([[1, 1, 0, 2], [1, 1, 0, 0]], bg=0)
        res = isolate_largest(grid)
        assert res.get(0, 0) == 1
        assert res.get(0, 3) == 0

    def test_center_content(self):
        grid = make_grid([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bg=0)
        res = center_content(grid)
        # Content (2x2 square) should now be centered in 4x4 canvas
        assert res.get(1, 1) == 1
        assert res.get(1, 2) == 1


class TestTTASymmetries:
    def test_transform_and_invert(self):
        grid = make_grid([[1, 2, 3], [4, 5, 6]])
        for sym_id in range(8):
            trans = transform_grid(grid, sym_id)
            inv = invert_grid(trans, sym_id)
            assert inv == grid, f"D4 symmetry {sym_id} roundtrip failed"
