"""
Unit tests for D4DihedralGroup and TTAVotingConsensus.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.tta_consensus import D4DihedralGroup, TTAVotingConsensus


class TestTTAConsensus:
    def test_d4_transform_roundtrip(self):
        # 3x4 asymmetric grid
        arr = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2]
        ])
        grid = ArcGrid(pixels=arr, background=0)

        for sym_id in range(8):
            transformed = D4DihedralGroup.apply(grid, sym_id)
            inverted = D4DihedralGroup.invert(transformed, sym_id)
            assert inverted == grid, f"D4 roundtrip failed for sym_id={sym_id}"

    def test_voting_consensus_majority(self):
        g1 = ArcGrid(pixels=np.ones((3, 3), dtype=int), background=0)
        g2 = ArcGrid(pixels=np.zeros((3, 3), dtype=int), background=0)

        predictions = [(g1, 0.9), (g1, 0.8), (g2, 0.4)]
        consensus_grid, conf = TTAVotingConsensus.aggregate_predictions(predictions)

        assert consensus_grid == g1
        assert conf > 0.5
