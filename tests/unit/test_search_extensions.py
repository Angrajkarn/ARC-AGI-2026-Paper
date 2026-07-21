"""
Unit tests for NeuralSearchPrior and MultiDimensionalHeuristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural_heuristic import MultiDimensionalHeuristic, NeuralSearchPrior


class TestSearchExtensions:
    def test_heuristic_exact_match(self):
        arr = np.array([[1, 2], [3, 4]])
        grid = ArcGrid(pixels=arr, background=0)

        evaluator = MultiDimensionalHeuristic()
        score = evaluator.evaluate(grid, grid)
        assert score.total_score == 1.0
        assert score.iou_score == 1.0
        assert score.entropy_similarity == 1.0

    def test_heuristic_partial_match(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [3, 0]])
        g1 = ArcGrid(pixels=arr1, background=0)
        g2 = ArcGrid(pixels=arr2, background=0)

        evaluator = MultiDimensionalHeuristic()
        score = evaluator.evaluate(g1, g2)
        assert 0.0 < score.total_score < 1.0
        assert 0.0 < score.iou_score < 1.0

    def test_neural_search_priors_prob_sum(self):
        g1 = ArcGrid(pixels=np.ones((4, 4), dtype=int), background=0)
        g2 = ArcGrid(pixels=np.ones((4, 4), dtype=int), background=0)

        prior_model = NeuralSearchPrior()
        priors = prior_model.predict_priors(g1, g2)

        assert len(priors) > 0
        assert pytest.approx(sum(priors.values()), rel=1e-5) == 1.0
        assert priors["rotate_90"] > 0
