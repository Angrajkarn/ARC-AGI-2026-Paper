"""
SceneGraph — converts an ARC grid into a structured object graph.

Each node is an ArcObject.  Edges encode spatial relationships:
  - ADJACENT   : objects share a border pixel
  - CONTAINS   : bounding box of B is inside bounding box of A
  - TOUCHES    : objects share a corner pixel (diagonal)
  - DISTANCE   : Euclidean centroid distance (always added)
  - SAME_COLOR : objects have the same colour
  - SAME_SHAPE : objects have identical pixel patterns (congruent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.core.objects.arc_object import ArcObject


class EdgeRelation(Enum):
    ADJACENT = auto()
    CONTAINS = auto()
    TOUCHES = auto()
    DISTANCE = auto()
    SAME_COLOR = auto()
    SAME_SHAPE = auto()


@dataclass
class GraphEdge:
    """A directed edge between two objects in the scene graph."""

    src_id: int
    dst_id: int
    relation: EdgeRelation
    properties: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Edge({self.src_id}→{self.dst_id}, {self.relation.name})"


class SceneGraph:
    """Object-centric scene graph for an ARC grid.

    Wraps a NetworkX DiGraph where:
      - Nodes are ArcObject instances (keyed by object_id).
      - Edges carry EdgeRelation labels and numeric/boolean properties.

    Usage::

        detector = ObjectDetector()
        objects = detector.detect(grid)
        graph = SceneGraph.build_from_objects(objects, grid)
        neighbours = graph.query_neighbors(obj_id=0, relation=EdgeRelation.ADJACENT)
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._objects: Dict[int, ArcObject] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_objects(
        cls,
        objects: List[ArcObject],
        grid_height: int,
        grid_width: int,
    ) -> "SceneGraph":
        """Build a scene graph from a list of detected objects.

        Args:
            objects:      List of ArcObject instances.
            grid_height:  Height of the source grid (for context).
            grid_width:   Width of the source grid.

        Returns:
            Populated SceneGraph instance.
        """
        sg = cls()
        sg._graph.graph["grid_height"] = grid_height
        sg._graph.graph["grid_width"] = grid_width

        for obj in objects:
            sg._graph.add_node(obj.object_id, object=obj)
            sg._objects[obj.object_id] = obj

        # Build edges between all pairs
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i == j:
                    continue
                sg._add_relationship_edges(a, b)

        return sg

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        obj_id: int,
        relation: Optional[EdgeRelation] = None,
    ) -> List[ArcObject]:
        """Return objects reachable from *obj_id* via a given relation.

        Args:
            obj_id:   Source object id.
            relation: If None, return all neighbours.

        Returns:
            List of neighbouring ArcObject instances.
        """
        neighbors = []
        for _, dst, data in self._graph.out_edges(obj_id, data=True):
            if relation is None or data.get("relation") == relation:
                neighbors.append(self._objects[dst])
        return neighbors

    def get_object(self, obj_id: int) -> ArcObject:
        return self._objects[obj_id]

    def all_objects(self) -> List[ArcObject]:
        return list(self._objects.values())

    def all_edges(self) -> List[GraphEdge]:
        edges = []
        for src, dst, data in self._graph.edges(data=True):
            edges.append(
                GraphEdge(
                    src_id=src,
                    dst_id=dst,
                    relation=data["relation"],
                    properties={k: v for k, v in data.items() if k != "relation"},
                )
            )
        return edges

    def objects_by_color(self, color: int) -> List[ArcObject]:
        return [o for o in self._objects.values() if o.color == color]

    def largest_object(self) -> Optional[ArcObject]:
        if not self._objects:
            return None
        return max(self._objects.values(), key=lambda o: o.area)

    def smallest_object(self) -> Optional[ArcObject]:
        if not self._objects:
            return None
        return min(self._objects.values(), key=lambda o: o.area)

    def __len__(self) -> int:
        return len(self._objects)

    def to_dict(self) -> Dict:
        """Serialise the scene graph to a plain dict (for logging/debugging)."""
        return {
            "nodes": [
                {
                    "id": obj.object_id,
                    "color": obj.color,
                    "area": obj.area,
                    "bbox": {
                        "row_min": obj.bounding_box.row_min,
                        "row_max": obj.bounding_box.row_max,
                        "col_min": obj.bounding_box.col_min,
                        "col_max": obj.bounding_box.col_max,
                    },
                    "tags": obj.tags,
                }
                for obj in self._objects.values()
            ],
            "edges": [
                {
                    "src": e.src_id,
                    "dst": e.dst_id,
                    "relation": e.relation.name,
                    **e.properties,
                }
                for e in self.all_edges()
            ],
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _add_relationship_edges(self, a: ArcObject, b: ArcObject) -> None:
        """Compute and add all relevant edges from a → b."""
        # DISTANCE (always)
        dist = a.distance_to(b)
        self._graph.add_edge(
            a.object_id,
            b.object_id,
            relation=EdgeRelation.DISTANCE,
            distance=round(dist, 3),
        )

        # ADJACENT (4-connected)
        if a.is_adjacent_to(b, diagonal=False):
            self._graph.add_edge(
                a.object_id, b.object_id, relation=EdgeRelation.ADJACENT
            )

        # TOUCHES (8-connected includes diagonal)
        elif a.is_adjacent_to(b, diagonal=True):
            self._graph.add_edge(
                a.object_id, b.object_id, relation=EdgeRelation.TOUCHES
            )

        # CONTAINS (a's bbox contains b's bbox)
        if a.contains_object(b):
            self._graph.add_edge(
                a.object_id, b.object_id, relation=EdgeRelation.CONTAINS
            )

        # SAME_COLOR
        if a.color == b.color:
            self._graph.add_edge(
                a.object_id, b.object_id, relation=EdgeRelation.SAME_COLOR
            )

        # SAME_SHAPE (identical pixel pattern after translation)
        if a.area == b.area and self._same_shape(a, b):
            self._graph.add_edge(
                a.object_id, b.object_id, relation=EdgeRelation.SAME_SHAPE
            )

    @staticmethod
    def _same_shape(a: ArcObject, b: ArcObject) -> bool:
        """Check if two objects have congruent shapes (translation-invariant)."""
        if a.area != b.area:
            return False
        # Normalise both to top-left = (0, 0)
        a_local = frozenset(
            (r - a.bounding_box.row_min, c - a.bounding_box.col_min)
            for r, c in a.pixels
        )
        b_local = frozenset(
            (r - b.bounding_box.row_min, c - b.bounding_box.col_min)
            for r, c in b.pixels
        )
        return a_local == b_local


__all__ = ["SceneGraph", "GraphEdge", "EdgeRelation"]
