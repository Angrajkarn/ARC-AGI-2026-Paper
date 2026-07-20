"""
Test-Time Augmentation (TTA) — Dihedral Group D4 symmetry search wrapper.

Applies 8 symmetry transformations to training pairs:
  - 0: Identity
  - 1: Rotate 90°
  - 2: Rotate 180°
  - 3: Rotate 270°
  - 4: Mirror Horizontal
  - 5: Mirror Vertical
  - 6: Transpose (Rotate 90° + Mirror H)
  - 7: Anti-transpose (Rotate 90° + Mirror V)

Runs search on transformed pairs and inverts predicted test outputs back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLProgram
from src.search.search_engine import SearchEngine, SearchResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Dihedral group D4 symmetry transformations & inverses
def transform_grid(grid: ArcGrid, sym_id: int) -> ArcGrid:
    """Apply symmetry transformation sym_id (0..7) to grid."""
    arr = grid.pixels.copy()
    if sym_id == 0:
        pass
    elif sym_id == 1:
        arr = np.rot90(arr, k=-1)  # 90 deg CW
    elif sym_id == 2:
        arr = np.rot90(arr, k=-2)  # 180 deg
    elif sym_id == 3:
        arr = np.rot90(arr, k=-3)  # 270 deg CW
    elif sym_id == 4:
        arr = np.fliplr(arr)       # Mirror H
    elif sym_id == 5:
        arr = np.flipud(arr)       # Mirror V
    elif sym_id == 6:
        arr = arr.T                # Transpose
    elif sym_id == 7:
        arr = np.rot90(np.fliplr(arr), k=1)  # Anti-transpose
    return ArcGrid(pixels=arr, background=grid.background)


def invert_grid(grid: ArcGrid, sym_id: int) -> ArcGrid:
    """Invert symmetry transformation sym_id (0..7) on grid."""
    # Inverse map for D4 symmetries
    inverses = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    inv_id = inverses[sym_id]
    return transform_grid(grid, inv_id)


class TTASearchWrapper:
    """Run search engine over 8 D4 dihedral symmetry transformations."""

    def __init__(
        self,
        base_search_engine: SearchEngine,
        num_symmetries: int = 4,  # default to top 4 symmetries for speed
    ) -> None:
        self.engine = base_search_engine
        self.num_symmetries = min(num_symmetries, 8)
        self._executor = DSLExecutor()

    def search_with_tta(
        self,
        hypotheses: List[Any],
        pairs: List[Dict],
        test_input: ArcGrid,
        time_budget: float = 60.0,
    ) -> List[Tuple[ArcGrid, float, str]]:
        """Run search across D4 symmetries.

        Args:
            hypotheses:  Seed hypotheses.
            pairs:       Training pairs.
            test_input:  Test input ArcGrid.
            time_budget: Total time budget in seconds.

        Returns:
            List of (predicted_output_grid, confidence, method_name) tuples.
        """
        t_start = time.perf_counter()
        results: List[Tuple[ArcGrid, float, str]] = []
        per_sym_budget = time_budget / self.num_symmetries

        for sym_id in range(self.num_symmetries):
            if time.perf_counter() - t_start > time_budget:
                break

            # Transform training pairs
            trans_pairs = [
                {
                    "input": transform_grid(p["input"], sym_id),
                    "output": transform_grid(p["output"], sym_id),
                }
                for p in pairs
            ]

            trans_test = transform_grid(test_input, sym_id)

            try:
                search_res = self.engine.search(hypotheses, trans_pairs, time_budget=per_sym_budget)
                if search_res.best_program is not None:
                    trans_pred, _ = self._executor.execute(search_res.best_program, trans_test)
                    if trans_pred is not None:
                        # Invert prediction back to original frame
                        orig_pred = invert_grid(trans_pred, sym_id)
                        conf = search_res.best_score
                        label = f"TTA_sym_{sym_id}_{search_res.algorithm}"
                        results.append((orig_pred, conf, label))
                        if search_res.found_perfect:
                            logger.info(f"TTA sym {sym_id}: found perfect program!")
                            break
            except Exception as e:
                logger.warning(f"TTA sym {sym_id} failed: {e}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results


__all__ = ["TTASearchWrapper", "transform_grid", "invert_grid"]
