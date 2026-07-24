"""
Unit tests for SkeletonConnectivityValidator.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_connectivity import SkeletonConnectivityValidator


class TestSkeletonConnectivityValidator:
    def test_is_connected_true(self):
        # A single connected diagonal line
        skeleton = {(0, 0), (1, 1), (2, 2)}
        assert SkeletonConnectivityValidator.is_connected(skeleton)

    def test_is_connected_false(self):
        # Two disconnected points
        skeleton = {(0, 0), (2, 2)}
        assert not SkeletonConnectivityValidator.is_connected(skeleton)
