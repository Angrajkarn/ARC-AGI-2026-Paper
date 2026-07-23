"""
Unit tests for AutonomousScientist.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.meta_learning.autonomous_scientist import AutonomousScientist


class TestAutonomousScientist:
    def test_analyze_failure_dim_mismatch(self):
        pred = ArcGrid(pixels=np.array([[1, 2]]), background=0)
        expected = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)

        analysis = AutonomousScientist.analyze_failure(pred, expected)
        assert analysis["failure_mode"] == "dimensions_mismatch"
        assert "resizing" in analysis["recommended_fix"]

    def test_analyze_failure_color_shift(self):
        pred = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        # Shift pixel 2 to 9
        expected = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        analysis = AutonomousScientist.analyze_failure(pred, expected)
        assert analysis["failure_mode"] == "color_shift"
        assert "add_color_mapping" in analysis["recommended_fix"]
