"""
HypergraphUnifier — constructs multi-edge relational hypergraphs over grid objects for relational pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


@dataclass
class HyperEdge:
    nodes: Tuple[int, ...]
    relation_type: str


class HypergraphUnifier:
    """Constructs multi-edge hypergraphs over grid objects to unify relational layouts."""

    def __init__(self) -> None:
        self.detector = ObjectDetector()

    def build_hypergraph(self, grid: ArcGrid) -> List[HyperEdge]:
        """Constructs list of hyperedges connecting grid objects based on collective relations."""
        objects = self.detector.detect(grid)
        hyperedges: List[HyperEdge] = []

        if len(objects) < 2:
            return hyperedges

        # Relation 1: Alignment Hyperedge (objects sharing the same row/col coordinate alignment)
        for r_align in range(grid.height):
            aligned_ids = tuple(
                obj.object_id for obj in objects if obj.bounding_box.row_min == r_align
            )
            if len(aligned_ids) >= 2:
                hyperedges.append(HyperEdge(nodes=aligned_ids, relation_type="row_alignment"))

        # Relation 2: Color Group Hyperedge (objects sharing the same non-background color)
        for c in range(10):
            if c == grid.background:
                continue
            same_color_ids = tuple(obj.object_id for obj in objects if obj.color == c)
            if len(same_color_ids) >= 2:
                hyperedges.append(HyperEdge(nodes=same_color_ids, relation_type="color_group"))

        return hyperedges
