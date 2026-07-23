"""
GraphMatcher — Computes node correspondence alignment between grid scene graphs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.graphs.scene_graph import SceneGraph


class GraphMatcher:
    """Finds correspondence matches between nodes of two Arc grids based on object traits."""

    @staticmethod
    def align_graphs(graph_a: SceneGraph, graph_b: SceneGraph) -> Dict[int, int]:
        """Aligns graph_a nodes to graph_b nodes by minimizing feature distance mapping."""
        objs_a = graph_a.all_objects()
        objs_b = graph_b.all_objects()

        alignment: Dict[int, int] = {}
        if not objs_a or not objs_b:
            return alignment

        # Find best color and area match for each node in a
        for a in objs_a:
            best_match_id = -1
            best_cost = float("inf")

            for b in objs_b:
                # Cost function: color difference + area size difference
                color_cost = 0.0 if a.color == b.color else 1.0
                size_cost = abs(a.area - b.area) / max(1, a.area)
                total_cost = color_cost + size_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_match_id = b.object_id

            if best_match_id != -1:
                alignment[a.object_id] = best_match_id

        return alignment
