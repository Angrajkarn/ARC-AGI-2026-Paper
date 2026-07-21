"""
CompositeObjectDetector — Groups adjacent connected components of differing colors into
multi-color composite objects (frames, containers, composite shapes) and hierarchical groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject, BoundingBox
from src.core.objects.detector import ObjectDetector


@dataclass
class CompositeObject:
    """Represents a composite multi-color object composed of multiple ArcObjects."""

    composite_id: int
    components: List[ArcObject]
    bounding_box: BoundingBox
    colors: Set[int]
    pixels: Set[Tuple[int, int]] = field(default_factory=set)

    @classmethod
    def from_components(cls, composite_id: int, components: List[ArcObject]) -> CompositeObject:
        all_pixels = set()
        all_colors = set()
        r_min, r_max, c_min, c_max = 999, -1, 999, -1

        for comp in components:
            all_pixels.update(comp.pixels)
            all_colors.add(comp.color)
            r_min = min(r_min, comp.bounding_box.row_min)
            r_max = max(r_max, comp.bounding_box.row_max)
            c_min = min(c_min, comp.bounding_box.col_min)
            c_max = max(c_max, comp.bounding_box.col_max)

        bbox = BoundingBox(row_min=r_min, row_max=r_max, col_min=c_min, col_max=c_max)
        return cls(
            composite_id=composite_id,
            components=components,
            bounding_box=bbox,
            colors=all_colors,
            pixels=all_pixels,
        )


class CompositeObjectDetector:
    """Detects multi-color composite structures and frames."""

    def __init__(self, max_touch_distance: int = 1) -> None:
        self.detector = ObjectDetector()
        self.max_touch_distance = max_touch_distance

    def detect_composite_objects(self, grid: ArcGrid) -> List[CompositeObject]:
        """Detects and returns all composite multi-color objects in the grid."""
        single_objects = self.detector.detect(grid)
        if not single_objects:
            return []

        # Graph of touching objects
        n = len(single_objects)
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if single_objects[i].is_adjacent_to(single_objects[j]):
                    adj[i].add(j)
                    adj[j].add(i)

        # Connected components over touching objects
        visited = set()
        composites: List[CompositeObject] = []
        comp_id = 0

        for i in range(n):
            if i not in visited:
                group = []
                queue = [i]
                visited.add(i)

                while queue:
                    curr = queue.pop(0)
                    group.append(single_objects[curr])
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)

                composite = CompositeObject.from_components(comp_id, group)
                composites.append(composite)
                comp_id += 1

        return composites
