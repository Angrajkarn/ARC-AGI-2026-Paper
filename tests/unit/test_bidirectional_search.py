"""
Unit tests for BidirectionalSearch.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.search.bidirectional_search import BidirectionalSearch


class TestBidirectionalSearch:
    def test_search_midpoint_none(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[5, 6], [7, 8]]), background=0)

        searcher = BidirectionalSearch()
        prog = searcher.search_midpoint([{"input": g1, "output": g2}])
        assert prog is None
