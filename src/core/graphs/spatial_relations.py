"""
SpatialRelationGraph — Computes fine-grained spatial relationships between objects,
including pixel containment/enclosure, directional relations (above, below, left, right),
and alignment properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject


class SpatialRelationType(Enum):
    ABOVE = auto()
    BELOW = auto()
    LEFT_OF = auto()
    RIGHT_OF = auto()
    ENCLOSES = auto()
    INSIDE = auto()
    TOUCHING = auto()
    HORIZONTALLY_ALIGNED = auto()
    VERTICALLY_ALIGNED = auto()


class SpatialRelationGraph:
    """Graph representation specialized for spatial layout and enclosure analysis."""

    def __init__(self, objects: List[ArcObject], grid: ArcGrid) -> None:
        self.objects = {obj.object_id: obj for obj in objects}
        self.grid = grid
        self.nx_graph = nx.DiGraph()

        for obj in objects:
            self.nx_graph.add_node(obj.object_id, object=obj)

        self._compute_relations()

    def _add_relation(self, src_id: int, dst_id: int, relation: SpatialRelationType) -> None:
        if not self.nx_graph.has_edge(src_id, dst_id):
            self.nx_graph.add_edge(src_id, dst_id, relations=set())
        self.nx_graph[src_id][dst_id]["relations"].add(relation)

    def _compute_relations(self) -> None:
        obj_list = list(self.objects.values())
        n = len(obj_list)

        for i in range(n):
            o1 = obj_list[i]
            for j in range(i + 1, n):
                o2 = obj_list[j]
                self._compute_pair_relations(o1, o2)

    def _compute_pair_relations(self, o1: ArcObject, o2: ArcObject) -> None:
        id1, id2 = o1.object_id, o2.object_id
        bbox1, bbox2 = o1.bounding_box, o2.bounding_box
        r1_min, c1_min, r1_max, c1_max = bbox1.row_min, bbox1.col_min, bbox1.row_max, bbox1.col_max
        r2_min, c2_min, r2_max, c2_max = bbox2.row_min, bbox2.col_min, bbox2.row_max, bbox2.col_max

        # Center points
        ctr1_r, ctr1_c = bbox1.center
        ctr2_r, ctr2_c = bbox2.center

        # Alignment
        if abs(ctr1_r - ctr2_r) <= 1.0:
            self._add_relation(id1, id2, SpatialRelationType.HORIZONTALLY_ALIGNED)
            self._add_relation(id2, id1, SpatialRelationType.HORIZONTALLY_ALIGNED)

        if abs(ctr1_c - ctr2_c) <= 1.0:
            self._add_relation(id1, id2, SpatialRelationType.VERTICALLY_ALIGNED)
            self._add_relation(id2, id1, SpatialRelationType.VERTICALLY_ALIGNED)

        # Directional relations
        if r1_max < r2_min:
            self._add_relation(id1, id2, SpatialRelationType.ABOVE)
            self._add_relation(id2, id1, SpatialRelationType.BELOW)
        elif r2_max < r1_min:
            self._add_relation(id2, id1, SpatialRelationType.ABOVE)
            self._add_relation(id1, id2, SpatialRelationType.BELOW)

        if c1_max < c2_min:
            self._add_relation(id1, id2, SpatialRelationType.LEFT_OF)
            self._add_relation(id2, id1, SpatialRelationType.RIGHT_OF)
        elif c2_max < c1_min:
            self._add_relation(id2, id1, SpatialRelationType.LEFT_OF)
            self._add_relation(id1, id2, SpatialRelationType.RIGHT_OF)

        # Enclosure / Containment
        if r1_min <= r2_min and r1_max >= r2_max and c1_min <= c2_min and c1_max >= c2_max:
            self._add_relation(id1, id2, SpatialRelationType.ENCLOSES)
            self._add_relation(id2, id1, SpatialRelationType.INSIDE)
        elif r2_min <= r1_min and r2_max >= r1_max and c2_min <= c1_min and c2_max >= c1_max:
            self._add_relation(id2, id1, SpatialRelationType.ENCLOSES)
            self._add_relation(id1, id2, SpatialRelationType.INSIDE)

        # Touch check
        if o1.is_adjacent_to(o2):
            self._add_relation(id1, id2, SpatialRelationType.TOUCHING)
            self._add_relation(id2, id1, SpatialRelationType.TOUCHING)

    def query_objects_with_relation(self, obj_id: int, relation: SpatialRelationType) -> List[ArcObject]:
        """Returns all ArcObjects that stand in specified relation to *obj_id*."""
        res = []
        if obj_id not in self.nx_graph:
            return res
        for u, v, data in self.nx_graph.out_edges(obj_id, data=True):
            relations = data.get("relations", set())
            if relation in relations and v in self.objects:
                res.append(self.objects[v])
        return res

    def get_enclosed_objects(self, container_obj_id: int) -> List[ArcObject]:
        """Returns objects enclosed inside container_obj_id."""
        return self.query_objects_with_relation(container_obj_id, SpatialRelationType.ENCLOSES)
