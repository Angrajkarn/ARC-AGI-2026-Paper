"""
Unit tests for SkeletonEulerSignature.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_euler import SkeletonEulerSignature


class TestSkeletonEulerSignature:
    def test_get_euler_characteristic_line(self):
        # A simple linear line: (0, 0) - (0, 1) - (0, 2)
        # V = 3, E = 2 -> V - E = 1
        skeleton = {(0, 0), (0, 1), (0, 2)}
        chi = SkeletonEulerSignature.get_euler_characteristic(skeleton)
        assert chi == 1

    def test_get_euler_characteristic_loop(self):
        # A 2x2 loop box: (0, 0) - (0, 1) - (1, 1) - (1, 0) - loop back
        # V = 4, edges: 4 neighbors, plus maybe diagonal overlaps. Let's trace carefully:
        # Node (0, 0) connects to (0, 1), (1, 0), and (1, 1) if 8-connected.
        # Let's check with vertical/horizontal neighbors loop:
        skeleton = {(0, 0), (0, 1), (1, 0), (1, 1)}
        # Connections:
        # (0, 0) to (0, 1) [0, 1], to (1, 0) [1, 0], to (1, 1) [1, 1]
        # (0, 1) to (1, 1) [1, 0], to (1, 0) [1, -1]
        # (1, 0) to (1, 1) [0, 1]
        # Total unique edges = 6 (fully connected K_4 graph)
        # V = 4, E = 6 -> chi = 4 - 6 = -2
        chi = SkeletonEulerSignature.get_euler_characteristic(skeleton)
        assert chi == -2
