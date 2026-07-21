"""
Unit tests for ARCDashboard.
"""

from __future__ import annotations

import numpy as np
import pytest
from rich.panel import Panel
from rich.table import Table

from src.core.grid.grid import ArcGrid
from src.ui.dashboard import ARCDashboard


class TestARCDashboard:
    def test_render_summary(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        dash = ARCDashboard()

        panel = dash.render_task_summary("007bbfb7", [{"input": grid, "output": grid}])
        assert isinstance(panel, Panel)

    def test_render_objects_table(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        dash = ARCDashboard()

        table = dash.render_objects_table(grid)
        assert isinstance(table, Table)
