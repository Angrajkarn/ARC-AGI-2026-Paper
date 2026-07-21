"""
Unit tests for InvariantProofVerifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.verifier.proof_verifier import InvariantProofVerifier


class TestProofVerifier:
    def test_valid_grid(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        report = InvariantProofVerifier.verify(grid)
        assert report.is_valid is True
        assert report.colors_valid is True
        assert report.dimensions_valid is True

    def test_invalid_colors(self):
        grid = ArcGrid(pixels=np.array([[1, 20], [3, 4]]), background=0)
        report = InvariantProofVerifier.verify(grid)
        assert report.is_valid is False
        assert report.colors_valid is False
