"""
Unit tests for SkeletonBettiBoundaryEstimator.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_betti_boundary import SkeletonBettiBoundaryEstimator


class TestSkeletonBettiBoundaryEstimator:
    def test_compute_betti_boundary_persistence_empty(self):
        score = SkeletonBettiBoundaryEstimator.compute_betti_boundary_persistence(set())
        assert score == 0.0

    def test_compute_betti_boundary_persistence_box(self):
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        score = SkeletonBettiBoundaryEstimator.compute_betti_boundary_persistence(pixels)
        assert score > 0.0
