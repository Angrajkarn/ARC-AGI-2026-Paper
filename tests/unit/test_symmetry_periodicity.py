"""
Unit tests for SymmetryAnalyzer, PeriodicityDetector, and ColorPaletteTransfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.features.symmetry_periodicity import ColorPaletteTransfer, PeriodicityDetector, SymmetryAnalyzer


class TestSymmetryPeriodicity:
    def test_symmetry_analyzer(self):
        # Perfectly symmetric 3x3 grid
        arr = np.array([
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1]
        ])
        grid = ArcGrid(pixels=arr, background=0)

        report = SymmetryAnalyzer.analyze(grid)
        assert report.horizontal_score == 1.0
        assert report.vertical_score == 1.0
        assert report.main_diagonal_score == 1.0
        assert report.rotational_90_score == 1.0

    def test_periodicity_detector(self):
        # 4x4 repeating 2x2 tile
        tile = np.array([[1, 2], [3, 4]])
        arr = np.tile(tile, (2, 2))
        grid = ArcGrid(pixels=arr, background=0)

        ph, pw = PeriodicityDetector.detect_period(grid)
        assert ph == 2
        assert pw == 2

    def test_color_palette_transfer(self):
        src = ArcGrid(pixels=np.array([[1, 1], [1, 0]]), background=0)
        tgt = ArcGrid(pixels=np.array([[5, 5], [5, 0]]), background=0)

        transferred = ColorPaletteTransfer.transfer(src, tgt)
        assert transferred.get(0, 0) == 5
