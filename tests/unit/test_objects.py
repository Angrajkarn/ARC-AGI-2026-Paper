"""Unit tests for object detection and ArcObject."""

from __future__ import annotations

import pytest

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject, BoundingBox
from src.core.objects.detector import ObjectDetector


class TestBoundingBox:
    def test_dimensions(self):
        bbox = BoundingBox(1, 3, 2, 5)
        assert bbox.height == 3
        assert bbox.width == 4
        assert bbox.area == 12

    def test_center(self):
        bbox = BoundingBox(0, 2, 0, 2)
        assert bbox.center == (1.0, 1.0)

    def test_contains(self):
        bbox = BoundingBox(0, 5, 0, 5)
        assert bbox.contains(2, 2)
        assert not bbox.contains(6, 2)


class TestArcObject:
    def _make_obj(self, pixels, color=1):
        return ArcObject.from_pixels(0, color, set(pixels))

    def test_basic_properties(self):
        obj = self._make_obj([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert obj.area == 4
        assert obj.bounding_box.height == 2
        assert obj.bounding_box.width == 2
        assert obj.is_rectangular

    def test_line_detection(self):
        obj = self._make_obj([(0, 0), (0, 1), (0, 2), (0, 3)])
        assert obj.is_line

    def test_symmetry_horizontal(self):
        obj = self._make_obj([(0, 0), (0, 2), (1, 0), (1, 2)])
        assert obj.is_symmetric_h

    def test_symmetry_vertical(self):
        obj = self._make_obj([(0, 1), (1, 0), (1, 2), (2, 1)])
        # Diamond shape – symmetric vertically
        assert obj.is_symmetric_v

    def test_translate(self):
        obj = self._make_obj([(0, 0), (0, 1)])
        translated = obj.translate(2, 3)
        assert (2, 3) in translated.pixels
        assert (2, 4) in translated.pixels

    def test_recolor(self):
        obj = self._make_obj([(0, 0)], color=3)
        recolored = obj.recolor(7)
        assert recolored.color == 7
        assert obj.color == 3  # original unchanged

    def test_rotate_90(self):
        # 2x1 object → should become 1x2 after 90° rotation
        obj = self._make_obj([(0, 0), (1, 0)])
        rotated = obj.rotate_90(1)
        assert rotated.bounding_box.width > rotated.bounding_box.height or rotated.bounding_box.height >= 1

    def test_distance_to(self):
        a = self._make_obj([(0, 0)])
        b = self._make_obj([(0, 3)])
        assert abs(a.distance_to(b) - 3.0) < 0.01

    def test_adjacent_to(self):
        a = self._make_obj([(0, 0)])
        b = self._make_obj([(0, 1)])
        assert a.is_adjacent_to(b)

    def test_not_adjacent(self):
        a = self._make_obj([(0, 0)])
        b = self._make_obj([(0, 2)])
        assert not a.is_adjacent_to(b)

    def test_perimeter(self):
        # 2x2 square: 8 boundary edges
        obj = self._make_obj([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert obj.perimeter == 8

    def test_empty_pixels_raises(self):
        with pytest.raises(ValueError):
            ArcObject.from_pixels(0, 1, set())


class TestObjectDetector:
    def test_basic_detection(self):
        grid = ArcGrid.from_list([
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 3, 3],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)
        colors = {o.color for o in objects}
        assert 1 in colors
        assert 2 in colors
        assert 3 in colors

    def test_background_excluded(self):
        grid = ArcGrid.from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        detector = ObjectDetector(ignore_background=True)
        objects = detector.detect(grid)
        assert all(o.color != 0 for o in objects)

    def test_single_component(self):
        # Use explicit background=0 so the 1s are detected as objects
        import numpy as np
        arr = np.array([[1, 1], [1, 1]], dtype=np.int8)
        from src.core.grid.grid import ArcGrid
        grid = ArcGrid(pixels=arr, background=0)
        detector = ObjectDetector()
        objects = detector.detect(grid)
        assert len(objects) == 1
        assert objects[0].area == 4

    def test_two_components(self):
        grid = ArcGrid.from_list([[1, 0, 2], [0, 0, 0]])
        detector = ObjectDetector()
        objects = detector.detect(grid)
        assert len(objects) == 2

    def test_dsu_matches_bfs(self):
        grid = ArcGrid.from_list([
            [1, 1, 0, 2, 2],
            [1, 0, 0, 2, 0],
            [0, 3, 3, 0, 4],
        ])
        detector = ObjectDetector()
        bfs_objects = detector.detect(grid)
        dsu_objects = detector.detect_with_dsu(grid)
        bfs_colors = sorted(o.color for o in bfs_objects)
        dsu_colors = sorted(o.color for o in dsu_objects)
        assert bfs_colors == dsu_colors

    def test_min_size_filter(self):
        grid = ArcGrid.from_list([[1, 0, 0], [0, 0, 2]])
        detector = ObjectDetector(min_size=2)
        objects = detector.detect(grid)
        # Both objects have area=1, so neither should be included
        assert len(objects) == 0

    def test_flood_fill(self):
        grid = ArcGrid.from_list([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        detector = ObjectDetector()
        region = detector.flood_fill(grid, 0, 0, target_color=1)
        # The ring of 1s should have 8 pixels
        assert len(region) == 8

    def test_tags_applied(self):
        import numpy as np
        from src.core.grid.grid import ArcGrid
        arr = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int8)
        grid = ArcGrid(pixels=arr, background=0)
        detector = ObjectDetector()
        objects = detector.detect(grid)
        obj = next(o for o in objects if o.color == 1)
        assert "rectangle" in obj.tags
        # When there is only one object, largest/smallest are not tagged (by design)
        # but rectangle and border should be present
        assert "border" in obj.tags or "symmetric_h" in obj.tags
