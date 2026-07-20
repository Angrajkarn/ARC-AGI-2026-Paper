"""
Feature Extractor — high-level feature vector for ARC grids.

Aggregates signals from geometry, topology, colour, and spatial analysis
into a single FeatureVector that is used by:
  - Rule discovery (comparing train pairs)
  - Search heuristics (scoring candidate programs)
  - LLM prompt generation (describing puzzles)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject
from src.core.objects.detector import ObjectDetector


# ---------------------------------------------------------------------------
# Sub-feature dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SymmetryFeatures:
    is_horizontally_symmetric: bool = False
    is_vertically_symmetric: bool = False
    is_rotationally_symmetric_180: bool = False
    has_diagonal_symmetry: bool = False


@dataclass
class ColorFeatures:
    num_colors: int = 0
    color_set: Set[int] = field(default_factory=set)
    dominant_color: int = 0
    color_histogram: Dict[int, int] = field(default_factory=dict)
    has_single_color: bool = False


@dataclass
class ObjectFeatures:
    num_objects: int = 0
    avg_area: float = 0.0
    max_area: int = 0
    min_area: int = 0
    all_same_color: bool = False
    all_same_shape: bool = False
    all_same_size: bool = False
    has_repeated_pattern: bool = False
    object_count_per_color: Dict[int, int] = field(default_factory=dict)


@dataclass
class SpatialFeatures:
    has_alignment_h: bool = False      # objects aligned horizontally
    has_alignment_v: bool = False      # objects aligned vertically
    has_grid_layout: bool = False      # objects on a regular grid
    has_border_frame: bool = False     # a frame / border around the grid
    has_enclosed_regions: bool = False  # holes detected
    connectivity: str = "disconnected"  # "single", "multi", "disconnected"


@dataclass
class TopologyFeatures:
    num_connected_components: int = 0
    euler_number: int = 0
    has_holes: bool = False
    has_lines: bool = False
    has_rectangles: bool = False


@dataclass
class FeatureVector:
    """Comprehensive feature vector for one ARC grid."""

    grid_height: int = 0
    grid_width: int = 0
    total_pixels: int = 0
    non_background_ratio: float = 0.0

    symmetry: SymmetryFeatures = field(default_factory=SymmetryFeatures)
    color: ColorFeatures = field(default_factory=ColorFeatures)
    objects: ObjectFeatures = field(default_factory=ObjectFeatures)
    spatial: SpatialFeatures = field(default_factory=SpatialFeatures)
    topology: TopologyFeatures = field(default_factory=TopologyFeatures)

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Flatten to a simple dict for logging or LLM prompting."""
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "non_background_ratio": round(self.non_background_ratio, 3),
            "num_colors": self.color.num_colors,
            "color_set": sorted(self.color.color_set),
            "num_objects": self.objects.num_objects,
            "max_area": self.objects.max_area,
            "is_h_symmetric": self.symmetry.is_horizontally_symmetric,
            "is_v_symmetric": self.symmetry.is_vertically_symmetric,
            "has_repeated_pattern": self.objects.has_repeated_pattern,
            "has_border_frame": self.spatial.has_border_frame,
            "has_holes": self.topology.has_holes,
            "has_lines": self.topology.has_lines,
            "has_rectangles": self.topology.has_rectangles,
        }


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract a FeatureVector from an ArcGrid.

    Usage::

        extractor = FeatureExtractor()
        features = extractor.extract(grid)
    """

    def __init__(self, connectivity: int = 4) -> None:
        self._detector = ObjectDetector(connectivity=connectivity)

    def extract(self, grid: ArcGrid) -> FeatureVector:
        """Compute all features for a grid.

        Args:
            grid: Input ArcGrid.

        Returns:
            Populated FeatureVector.
        """
        fv = FeatureVector(
            grid_height=grid.height,
            grid_width=grid.width,
            total_pixels=grid.num_pixels,
        )

        objects = self._detector.detect(grid)
        non_bg = int((grid.pixels != grid.background).sum())
        fv.non_background_ratio = non_bg / max(grid.num_pixels, 1)

        fv.symmetry = self._symmetry(grid)
        fv.color = self._color(grid)
        fv.objects = self._objects(objects)
        fv.spatial = self._spatial(objects, grid)
        fv.topology = self._topology(objects)

        return fv

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------

    def _symmetry(self, grid: ArcGrid) -> SymmetryFeatures:
        arr = grid.pixels
        return SymmetryFeatures(
            is_horizontally_symmetric=bool(np.array_equal(arr, arr[:, ::-1])),
            is_vertically_symmetric=bool(np.array_equal(arr, arr[::-1, :])),
            is_rotationally_symmetric_180=bool(np.array_equal(arr, arr[::-1, ::-1])),
            has_diagonal_symmetry=(
                arr.shape[0] == arr.shape[1]
                and bool(np.array_equal(arr, arr.T))
            ),
        )

    # ------------------------------------------------------------------
    # Colour
    # ------------------------------------------------------------------

    def _color(self, grid: ArcGrid) -> ColorFeatures:
        arr = grid.pixels
        values, counts = np.unique(arr, return_counts=True)
        histogram = {int(v): int(c) for v, c in zip(values, counts)}
        dominant = int(values[np.argmax(counts)])
        color_set = set(int(v) for v in values)
        return ColorFeatures(
            num_colors=len(color_set),
            color_set=color_set,
            dominant_color=dominant,
            color_histogram=histogram,
            has_single_color=(len(color_set) == 1),
        )

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------

    def _objects(self, objects: List[ArcObject]) -> ObjectFeatures:
        if not objects:
            return ObjectFeatures()

        areas = [o.area for o in objects]
        colors = [o.color for o in objects]
        shapes = [
            frozenset(
                (r - o.bounding_box.row_min, c - o.bounding_box.col_min)
                for r, c in o.pixels
            )
            for o in objects
        ]
        count_per_color: Dict[int, int] = {}
        for c in colors:
            count_per_color[c] = count_per_color.get(c, 0) + 1

        return ObjectFeatures(
            num_objects=len(objects),
            avg_area=sum(areas) / len(areas),
            max_area=max(areas),
            min_area=min(areas),
            all_same_color=(len(set(colors)) == 1),
            all_same_shape=(len(set(shapes)) == 1),
            all_same_size=(len(set(areas)) == 1),
            has_repeated_pattern=(max(count_per_color.values(), default=0) > 1),
            object_count_per_color=count_per_color,
        )

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def _spatial(self, objects: List[ArcObject], grid: ArcGrid) -> SpatialFeatures:
        if not objects:
            return SpatialFeatures()

        # Alignment: objects share the same row_min or col_min
        row_mins = [o.bounding_box.row_min for o in objects]
        col_mins = [o.bounding_box.col_min for o in objects]
        has_align_h = (len(set(row_mins)) < len(objects)) if len(objects) > 1 else False
        has_align_v = (len(set(col_mins)) < len(objects)) if len(objects) > 1 else False

        # Border frame: all edge pixels same non-background colour
        arr = grid.pixels
        H, W = arr.shape
        if H >= 3 and W >= 3:
            top = arr[0, :]
            bottom = arr[-1, :]
            left = arr[:, 0]
            right = arr[:, -1]
            edge_colors = set(np.unique(np.concatenate([top, bottom, left, right])).tolist())
            bg = grid.background
            has_border = (
                len(edge_colors) == 1
                and list(edge_colors)[0] != bg
                and bool((arr[1:-1, 1:-1] != list(edge_colors)[0]).any())
            )
        else:
            has_border = False

        has_enclosed = any(bool(o.holes) for o in objects)

        n = len(objects)
        if n == 1:
            connectivity = "single"
        elif n <= 5:
            connectivity = "multi"
        else:
            connectivity = "disconnected"

        return SpatialFeatures(
            has_alignment_h=has_align_h,
            has_alignment_v=has_align_v,
            has_grid_layout=self._detect_grid_layout(objects),
            has_border_frame=has_border,
            has_enclosed_regions=has_enclosed,
            connectivity=connectivity,
        )

    def _detect_grid_layout(self, objects: List[ArcObject]) -> bool:
        """Heuristic: do objects lie on a regular grid?"""
        if len(objects) < 4:
            return False
        rows = sorted({o.bounding_box.row_min for o in objects})
        cols = sorted({o.bounding_box.col_min for o in objects})
        if len(rows) < 2 or len(cols) < 2:
            return False
        row_gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        col_gaps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        row_regular = (max(row_gaps) - min(row_gaps)) <= 1
        col_regular = (max(col_gaps) - min(col_gaps)) <= 1
        return row_regular and col_regular

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _topology(self, objects: List[ArcObject]) -> TopologyFeatures:
        n_comp = len(objects)
        n_holes = sum(len(o.holes) for o in objects)
        has_lines = any("line" in o.tags for o in objects)
        has_rects = any("rectangle" in o.tags for o in objects)

        # Euler number approximation: connected_components - holes
        euler = n_comp - n_holes

        return TopologyFeatures(
            num_connected_components=n_comp,
            euler_number=euler,
            has_holes=n_holes > 0,
            has_lines=has_lines,
            has_rectangles=has_rects,
        )


__all__ = ["FeatureExtractor", "FeatureVector"]
