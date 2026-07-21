"""
Unit tests for SpatialRelationGraph.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graphs.spatial_relations import SpatialRelationGraph, SpatialRelationType
from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


class TestSpatialRelations:
    def test_directional_relations(self):
        # 10x10 grid with object 1 at top (1,1) and object 2 at bottom (8,8)
        arr = np.zeros((10, 10), dtype=int)
        arr[1, 1] = 1
        arr[8, 8] = 2
        grid = ArcGrid(pixels=arr, background=0)

        detector = ObjectDetector()
        objects = detector.detect(grid)
        graph = SpatialRelationGraph(objects, grid)

        # Object at (1,1) should be ABOVE and LEFT_OF object at (8,8)
        obj1 = [o for o in objects if o.color == 1][0]
        obj2 = [o for o in objects if o.color == 2][0]

        above_objs = graph.query_objects_with_relation(obj1.object_id, SpatialRelationType.ABOVE)
        assert obj2 in above_objs

        left_objs = graph.query_objects_with_relation(obj1.object_id, SpatialRelationType.LEFT_OF)
        assert obj2 in left_objs

    def test_enclosure_relation(self):
        # 7x7 grid with an outer square (color 2) enclosing an inner shape (color 3)
        arr = np.zeros((7, 7), dtype=int)
        # Outer box of color 2
        arr[1, 1:6] = 2
        arr[5, 1:6] = 2
        arr[1:6, 1] = 2
        arr[1:6, 5] = 2
        # Inner dot of color 3
        arr[3, 3] = 3
        grid = ArcGrid(pixels=arr, background=0)

        detector = ObjectDetector()
        objects = detector.detect(grid)
        graph = SpatialRelationGraph(objects, grid)

        outer_obj = [o for o in objects if o.color == 2][0]
        inner_obj = [o for o in objects if o.color == 3][0]

        enclosed = graph.get_enclosed_objects(outer_obj.object_id)
        assert inner_obj in enclosed
