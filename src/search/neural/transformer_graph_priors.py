"""
TransformerGraphPriors — scores transformations using self-attention over scene graph nodes.
"""

from __future__ import annotations

import numpy as np

from src.core.graphs.scene_graph import SceneGraph


class TransformerGraphPriors:
    """Computes action priority vectors utilizing self-attention keys/queries over scene graph nodes."""

    def __init__(self, embedding_dim: int = 4) -> None:
        self.embedding_dim = embedding_dim
        # Deterministically initialize projection matrices
        np.random.seed(101)
        self.w_q = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.w_k = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))

    def compute_prior_distribution(self, graph: SceneGraph) -> np.ndarray:
        """Calculates scalar attention weight average representation for candidate transformation selection."""
        objects = graph.all_objects()
        if not objects:
            return np.zeros(self.embedding_dim)

        # Build feature matrix: color, area, row_min, col_min
        features = []
        for obj in objects:
            features.append([
                obj.color,
                obj.area,
                obj.bounding_box.row_min,
                obj.bounding_box.col_min
            ])

        x = np.array(features, dtype=float)
        # Pad columns if dimensions are smaller than embedding_dim
        if x.shape[1] < self.embedding_dim:
            padding = np.zeros((x.shape[0], self.embedding_dim - x.shape[1]))
            x = np.hstack([x, padding])

        q = x @ self.w_q
        k = x @ self.w_k

        # Dot-product attention matrix
        scores = (q @ k.T) / np.sqrt(self.embedding_dim)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        return np.mean(weights, axis=0)
