"""
Unit tests for CompositeObjectDetector.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.objects.composite_detector import CompositeObjectDetector


class TestCompositeDetector:
    def test_multi_color_composite_detection(self):
        # 5x5 grid with touching pixels of color 1 and color 2
        arr = np.zeros((5, 5), dtype=int)
        arr[1, 1:3] = 1
        arr[1, 3] = 2
        grid = ArcGrid(pixels=arr, background=0)

        detector = CompositeObjectDetector()
        composites = detector.detect_composite_objects(grid)

        assert len(composites) == 1
        assert composites[0].colors == {1, 2}
        assert len(composites[0].components) == 2
