"""Unit tests for Phase 3 extensions: RuleTemplateSynthesizer, HeuristicEvaluator, and Subgrid primitives."""

from __future__ import annotations

import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.subgrid_primitives import (
    extract_repeating_tile,
    split_by_grid_lines,
    tile_to_fit,
)
from src.reasoning.rule_templates import RuleTemplateSynthesizer
from src.search.heuristic_evaluator import HeuristicEvaluator


def make_grid(data, bg=0):
    return ArcGrid.from_list(data, background=bg)


class TestRuleTemplateSynthesizer:
    def test_synthesize_crop_and_rotate(self):
        # Input: 3x3 grid with content in center -> Output: rotated content
        g_in = make_grid([[0, 0, 0], [0, 1, 2], [0, 0, 0]])
        g_out = make_grid([[1], [2]])  # cropped and rotated 90
        pairs = [{"input": g_in, "output": g_out}]

        synthesizer = RuleTemplateSynthesizer()
        candidates = synthesizer.synthesize_templates(pairs)
        assert len(candidates) >= 1
        assert candidates[0].confidence == 1.0


class TestHeuristicEvaluator:
    def test_exact_match_score(self):
        g = make_grid([[1, 2], [3, 4]])
        evaluator = HeuristicEvaluator()
        sim = evaluator.evaluate_similarity(g, g)
        assert sim.total_score == 1.0
        assert sim.pixel_similarity == 1.0

    def test_partial_pixel_match(self):
        g1 = make_grid([[1, 2], [3, 4]])
        g2 = make_grid([[1, 2], [3, 9]])  # 3 of 4 pixels match
        evaluator = HeuristicEvaluator()
        sim = evaluator.evaluate_similarity(g1, g2)
        assert 0.7 <= sim.pixel_similarity <= 0.8
        assert sim.total_score > 0.5

    def test_color_histogram_similarity(self):
        g1 = make_grid([[1, 1], [1, 1]])
        g2 = make_grid([[1, 1], [1, 2]])
        evaluator = HeuristicEvaluator()
        sim = evaluator.evaluate_similarity(g1, g2)
        assert sim.color_similarity > 0.8


class TestSubgridPrimitives:
    def test_extract_repeating_tile(self):
        # 4x4 grid consisting of 2x2 tile repeated
        tile_data = [[1, 2], [3, 4]]
        grid = make_grid([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        tile = extract_repeating_tile(grid)
        assert tile.to_list() == tile_data

    def test_tile_to_fit(self):
        tile = make_grid([[1, 2]])
        tiled = tile_to_fit(tile, target_height=2, target_width=4)
        assert tiled.to_list() == [[1, 2, 1, 2], [1, 2, 1, 2]]

    def test_split_by_grid_lines(self):
        # 3x3 cells partitioned by line_color 5
        grid = make_grid([
            [1, 5, 2],
            [5, 5, 5],
            [3, 5, 4],
        ])
        subgrids = split_by_grid_lines(grid, line_color=5)
        assert len(subgrids) == 4
        assert subgrids[0].to_list() == [[1]]
        assert subgrids[1].to_list() == [[2]]
