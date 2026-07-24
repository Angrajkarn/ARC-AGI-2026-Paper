"""
Unit tests for ScalePrior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.scale_prior import ScalePrior


class TestScalePrior:
    def test_estimate_scale_factor_double(self):
        # input is 2x2, output is 4x4
        inp = ArcGrid(pixels=np.zeros((2, 2), dtype=np.uint8), background=0)
        out = ArcGrid(pixels=np.zeros((4, 4), dtype=np.uint8), background=0)

        h_scale, w_scale = ScalePrior.estimate_scale_factor([(inp, out)])

        assert h_scale == 2.0
        assert w_scale == 2.0
