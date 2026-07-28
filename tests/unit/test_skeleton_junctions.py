"""
Unit tests for SkeletonJunctionAnalyzer.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_junctions import SkeletonJunctionAnalyzer


class TestSkeletonJunctionAnalyzer:
    def test_get_junctions_and_endpoints_line(self):
        # Line: (0, 0) - (0, 1) - (0, 2)
        # Endpoints: (0, 0) and (0, 2), Junctions: none
        skeleton = {(0, 0), (0, 1), (0, 2)}
        juncts, ends = SkeletonJunctionAnalyzer.get_junctions_and_endpoints(skeleton)

        assert len(juncts) == 0
        assert ends == {(0, 0), (0, 2)}

    def test_get_junctions_and_endpoints_t_junction(self):
        # T-junction shape: center is (1, 1), connected to (0, 1), (2, 1), (1, 2)
        skeleton = {(1, 1), (0, 1), (2, 1), (1, 2)}
        juncts, ends = SkeletonJunctionAnalyzer.get_junctions_and_endpoints(skeleton)

        assert juncts == {(1, 1)}
        assert len(ends) == 3
