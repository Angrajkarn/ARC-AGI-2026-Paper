"""
Unit tests for CoordTransformPrior.
"""

from __future__ import annotations

import pytest

from src.search.neural.coord_transform_prior import CoordTransformPrior


class TestCoordTransformPrior:
    def test_transform_coordinates_mirror_h(self):
        coords = [(0, 0), (1, 2)]
        transformed = CoordTransformPrior.transform_coordinates(
            coords, grid_height=3, grid_width=4, action="mirror_horizontal"
        )

        # (0,0) -> (0,3); (1,2) -> (1,1)
        assert transformed[0] == (0, 3)
        assert transformed[1] == (1, 1)

    def test_transform_coordinates_mirror_v(self):
        coords = [(0, 0), (1, 2)]
        transformed = CoordTransformPrior.transform_coordinates(
            coords, grid_height=3, grid_width=4, action="mirror_vertical"
        )

        # (0,0) -> (2,0); (1,2) -> (1,2)
        assert transformed[0] == (2, 0)
        assert transformed[1] == (1, 2)
