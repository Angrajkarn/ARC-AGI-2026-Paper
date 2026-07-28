"""
Unit tests for SkeletonHomologyMatcher.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_homology import SkeletonHomologyMatcher


class TestSkeletonHomologyMatcher:
    def test_compute_cycle_rank_line(self):
        # A simple line -> cycle rank is 0 (no loops)
        skeleton = {(0, 0), (0, 1), (0, 2)}
        rank = SkeletonHomologyMatcher.compute_cycle_rank(skeleton)
        assert rank == 0

    def test_compute_cycle_rank_loop(self):
        # A 2x2 grid (fully connected 4 nodes) -> 6 edges, 4 vertices, 1 component -> 6 - 4 + 1 = 3 cycles
        skeleton = {(0, 0), (0, 1), (1, 0), (1, 1)}
        rank = SkeletonHomologyMatcher.compute_cycle_rank(skeleton)
        assert rank == 3
