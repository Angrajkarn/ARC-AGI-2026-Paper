"""
Unit tests for ContourTracer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.contour_tracer import ContourTracer


class TestContourTracer:
    def test_trace_contour(self):
        # 3x3 grid, a solid 2x2 square in the top-left corner
        pixels = np.zeros((3, 3), dtype=np.uint8)
        pixels[0, 0] = 1
        pixels[0, 1] = 1
        pixels[1, 0] = 1
        pixels[1, 1] = 1

        grid = ArcGrid(pixels=pixels, background=0)
        object_pixels = {(0, 0), (0, 1), (1, 0), (1, 1)}

        contour = ContourTracer.trace_contour(grid, object_pixels)

        # All four points are on the perimeter/border
        assert len(contour) == 4
        assert (0, 0) in contour
        assert (1, 1) in contour
