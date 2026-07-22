"""
Unit tests for SelfCorrectionLoop.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.reflection.correction_loop import SelfCorrectionLoop


class TestCorrectionLoop:
    def test_compute_error_delta(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        delta = SelfCorrectionLoop.compute_error_delta(g1, g2)
        assert (delta == np.array([[0, 1], [0, 0]])).all()

    def test_apply_patch(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        patched = SelfCorrectionLoop.apply_patch(g1, g2, max_patches=1)
        assert patched.get(0, 1) == 9
        assert patched.get(0, 0) == 1
