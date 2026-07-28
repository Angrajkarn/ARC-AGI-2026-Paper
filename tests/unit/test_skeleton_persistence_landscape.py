"""
Unit tests for SkeletonPersistenceLandscapeMatcher.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_persistence_landscape import SkeletonPersistenceLandscapeMatcher


class TestSkeletonPersistenceLandscapeMatcher:
    def test_compute_landscape_vector_empty(self):
        vec = SkeletonPersistenceLandscapeMatcher.compute_landscape_vector(set(), num_bins=5)
        assert vec == [0.0] * 5

    def test_compute_landscape_vector_box(self):
        skeleton = {(0, 0), (0, 1), (1, 0), (1, 1)}
        vec = SkeletonPersistenceLandscapeMatcher.compute_landscape_vector(skeleton, num_bins=5)

        assert len(vec) == 5
        assert pytest.approx(sum(vec)) == 1.0
