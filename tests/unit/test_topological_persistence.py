"""
Unit tests for TopologicalPersistence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.topological_persistence import TopologicalPersistence


class TestTopologicalPersistence:
    def test_compute_persistence_diagram(self):
        # A simple grid with 2 isolated color segments of color 3
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1, 1] = 3
        pixels[3, 3] = 3
        grid = ArcGrid(pixels=pixels, background=0)

        barcodes = TopologicalPersistence.compute_persistence_diagram(grid)
        assert len(barcodes) == 1
        assert barcodes[0]["color"] == 3
        # Expected death corresponds to components count (2 components)
        assert barcodes[0]["death"] == 2.0
