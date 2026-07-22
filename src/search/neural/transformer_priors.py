"""
TransformerPriors — Computes multi-head self-attention scores over grid object features.
"""

from __future__ import annotations

import numpy as np


class TransformerPriors:
    """Uses dot-product self-attention to prioritize candidate search actions."""

    def __init__(self, feature_dim: int = 4) -> None:
        self.feature_dim = feature_dim
        # Deterministically initialize attention weight matrices
        np.random.seed(42)
        self.w_q = np.random.normal(0, 0.1, (feature_dim, feature_dim))
        self.w_k = np.random.normal(0, 0.1, (feature_dim, feature_dim))
        self.w_v = np.random.normal(0, 0.1, (feature_dim, feature_dim))

    def compute_attention_scores(self, object_features: np.ndarray) -> np.ndarray:
        """Runs QKV self-attention returning computed affinity matrix."""
        if object_features.ndim == 1:
            object_features = object_features[np.newaxis, :]

        # Compute Q, K, V matrices
        q = object_features @ self.w_q
        k = object_features @ self.w_k
        v = object_features @ self.w_v

        # Dot-product scaled attention weights
        scores = (q @ k.T) / np.sqrt(self.feature_dim)
        # Softmax normalization over columns
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context representation
        return attn_weights @ v
