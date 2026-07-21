"""
D4 Group Test-Time Augmentation (TTA) & Multi-Symmetry Voting Consensus Engine.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class D4DihedralGroup:
    """Represents the D4 Dihedral Symmetry Group (8 elements)."""

    INVERSES = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}

    @staticmethod
    def apply(grid: ArcGrid, sym_id: int) -> ArcGrid:
        """Applies D4 transformation sym_id (0..7) to ArcGrid."""
        arr = grid.pixels.copy()
        if sym_id == 0:
            pass
        elif sym_id == 1:
            arr = np.rot90(arr, k=-1)
        elif sym_id == 2:
            arr = np.rot90(arr, k=-2)
        elif sym_id == 3:
            arr = np.rot90(arr, k=-3)
        elif sym_id == 4:
            arr = np.fliplr(arr)
        elif sym_id == 5:
            arr = np.flipud(arr)
        elif sym_id == 6:
            arr = arr.T
        elif sym_id == 7:
            arr = np.rot90(np.fliplr(arr), k=1)
        return ArcGrid(pixels=arr, background=grid.background)

    @staticmethod
    def invert(grid: ArcGrid, sym_id: int) -> ArcGrid:
        """Applies inverse D4 transformation to ArcGrid."""
        inv_id = D4DihedralGroup.INVERSES.get(sym_id, 0)
        return D4DihedralGroup.apply(grid, inv_id)


class TTAVotingConsensus:
    """Computes majority-voting consensus across TTA predictions."""

    @staticmethod
    def aggregate_predictions(predictions: List[Tuple[ArcGrid, float]]) -> Tuple[ArcGrid, float]:
        """
        Aggregates multiple (ArcGrid, confidence_weight) pairs.
        Returns consensus ArcGrid and overall confidence score.
        """
        if not predictions:
            raise ValueError("Empty predictions list provided to TTAVotingConsensus")

        if len(predictions) == 1:
            return predictions[0][0], predictions[0][1]

        # Exact grid majority vote weighted by confidence
        grid_votes: Dict[str, float] = {}
        grid_map = {}

        for grid, weight in predictions:
            key = f"{grid.pixels.tobytes().hex()}_{grid.background}"
            grid_votes[key] = grid_votes.get(key, 0.0) + weight
            grid_map[key] = grid

        total_weight = sum(w for _, w in predictions)
        best_key = max(grid_votes.items(), key=lambda x: x[1])[0]

        consensus_grid = grid_map[best_key]
        consensus_confidence = grid_votes[best_key] / (total_weight + 1e-9)

        return consensus_grid, float(consensus_confidence)
