"""
Unit tests for generate_paper_figures script.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_paper_figures import (
    generate_paper_figures,
    grid_to_svg,
    grid_to_tikz,
)
from src.core.grid.grid import ArcGrid


class TestPaperFigures:
    def test_grid_to_svg(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        svg = grid_to_svg(grid)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_grid_to_tikz(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        tikz = grid_to_tikz(grid)
        assert r"\begin{tikzpicture}" in tikz
        assert r"\end{tikzpicture}" in tikz

    def test_generate_paper_figures(self):
        success = generate_paper_figures()
        assert success is True
