"""
Unit tests for ClipGridEncoder.
"""

from __future__ import annotations

import pytest

from src.search.neural.clip_grid_encoder import ClipGridEncoder


class TestClipGridEncoder:
    def test_compute_similarity_rotate(self):
        encoder = ClipGridEncoder()
        sim_correct = encoder.compute_similarity("rotate the blue object", "rotate_90")
        sim_incorrect = encoder.compute_similarity("rotate the blue object", "scale_2x")

        assert sim_correct > sim_incorrect
        assert sim_correct > 0.5
        assert sim_incorrect == 0.0

    def test_compute_similarity_mirror(self):
        encoder = ClipGridEncoder()
        sim = encoder.compute_similarity("mirror image vertically", "mirror_vertical")
        assert sim > 0.7
