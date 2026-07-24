"""
Unit tests for BettiNumberMatcher.
"""

from __future__ import annotations
import pytest

from src.core.graphs.betti_matcher import BettiNumberMatcher


class TestBettiNumberMatcher:
    def test_get_betti_numbers(self):
        # A hollow box (Betti-0 components = 1, Betti-1 loops/holes = 1)
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }

        b0, b1 = BettiNumberMatcher.get_betti_numbers(pixels)

        assert b0 == 1
        assert b1 == 1
