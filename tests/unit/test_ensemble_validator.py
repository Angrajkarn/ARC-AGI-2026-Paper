"""
Unit tests for MultiModelLLMValidator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.ensemble_validator import MultiModelLLMValidator


class TestEnsembleValidator:
    def test_get_consensus_majority(self):
        # Two identical grids, one different grid
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g3 = ArcGrid(pixels=np.array([[9, 9], [9, 9]]), background=0)

        consensus = MultiModelLLMValidator.get_consensus([g1, g2, g3])
        assert consensus is not None
        assert consensus.get(0, 0) == 1
