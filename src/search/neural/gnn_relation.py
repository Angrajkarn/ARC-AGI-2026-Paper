"""
GNNRelationEncoder — message passing relation encoder over grid object scene graphs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.graphs.scene_graph import SceneGraph


class GNNRelationEncoder:
    """Simulates GNN message passing over object node graphs to encode relational traits."""

    def __init__(self, node_feature_dim: int = 4, passes: int = 2) -> None:
        self.node_feature_dim = node_feature_dim
        self.passes = passes

    def encode_relations(self, scene_graph: SceneGraph) -> Dict[int, np.ndarray]:
        """Runs message passing on scene graph node features, returning updated embeddings."""
        objects = scene_graph.all_objects()
        edges = scene_graph.all_edges()

        # Initialize node feature vectors: [color, size, center_x, center_y]
        embeddings: Dict[int, np.ndarray] = {}
        for obj in objects:
            emb = np.array([
                obj.color,
                len(obj.pixels),
                obj.bounding_box.row_min,
                obj.bounding_box.col_min,
            ], dtype=float)
            embeddings[obj.object_id] = emb

        # Graph Message Passing iterations
        for _ in range(self.passes):
            next_embeddings: Dict[int, np.ndarray] = {}
            for obj in objects:
                node_id = obj.object_id
                # Aggregate incoming neighbor messages
                neighbor_embs = []
                for edge in edges:
                    if edge.dst_id == node_id and edge.src_id in embeddings:
                        neighbor_embs.append(embeddings[edge.src_id])
                    elif edge.src_id == node_id and edge.dst_id in embeddings:
                        neighbor_embs.append(embeddings[edge.dst_id])

                if neighbor_embs:
                    agg_message = np.mean(neighbor_embs, axis=0)
                    # Update node representation: 0.5 * current + 0.5 * incoming
                    next_embeddings[node_id] = 0.5 * embeddings[node_id] + 0.5 * agg_message
                else:
                    next_embeddings[node_id] = embeddings[node_id]
            embeddings = next_embeddings

        return embeddings
