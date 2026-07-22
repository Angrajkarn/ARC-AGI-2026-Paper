"""
Unit tests for TransformerPriors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.search.neural.transformer_priors import TransformerPriors


class TestTransformerPriors:
    def test_compute_attention_scores(self):
        # 3 objects, each represented by a 4-dimensional feature vector
        features = np.array([
            [1.0, 0.0, 3.0, 0.5],
            [0.2, 1.0, 0.0, 2.0],
            [1.5, 0.8, 1.2, 0.0],
        ])

        model = TransformerPriors(feature_dim=4)
        scores = model.compute_attention_scores(features)

        # Output context representation should match input feature count & dim shape
        assert scores.shape == (3, 4)
