"""
Unit tests for RelaxedProgramSearch.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.relaxed_search import RelaxedProgramSearch


class TestRelaxedSearch:
    def test_search_relaxed(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        cands = ["rotate_90", "mirror_horizontal", "fill_holes"]

        searcher = RelaxedProgramSearch(candidates=cands)
        results = searcher.search_relaxed([{"input": g1, "output": g1}])

        assert len(results) == 3
        # Weights should sum to 1.0 (approximately)
        total_weight = sum(w for _, w in results)
        assert pytest.approx(total_weight) == 1.0
        assert results[0][1] > results[2][1]  # Highest-scoring candidate has highest weight
