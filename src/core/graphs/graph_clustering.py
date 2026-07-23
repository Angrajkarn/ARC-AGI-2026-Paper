"""
GraphClustering — Unsupervised clustering of scene graph nodes.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.core.graphs.scene_graph import SceneGraph


class GraphClustering:
    """Groups scene graph object nodes based on shape properties and color attributes."""

    @staticmethod
    def cluster_objects(graph: SceneGraph) -> Dict[int, List[int]]:
        """Groups objects by exact area size. Returns mapping of cluster ID to object IDs."""
        objects = graph.all_objects()
        if not objects:
            return {}

        clusters: Dict[int, List[int]] = {}
        for idx, obj in enumerate(objects):
            area = obj.area
            found_cluster = False
            for cluster_key in list(clusters.keys()):
                if cluster_key == area:
                    clusters[cluster_key].append(idx)
                    found_cluster = True
                    break

            if not found_cluster:
                clusters[area] = [idx]

        return clusters
