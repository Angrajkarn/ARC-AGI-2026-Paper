"""
Unit tests for StreamlitARCApp helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.ui.app import StreamlitARCApp


class TestWebApp:
    def test_grid_to_html_table(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        app = StreamlitARCApp()

        html = app.grid_to_html_table(grid, cell_size=20)
        assert "<table" in html
        assert "#0074D9" in html  # Blue color hex
