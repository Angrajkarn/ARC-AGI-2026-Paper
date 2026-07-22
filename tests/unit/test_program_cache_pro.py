"""
Unit tests for SubtreeProgramCache.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.search.program_cache_pro import SubtreeProgramCache


class TestSubtreeProgramCache:
    def test_cache_hit_miss(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[5, 6], [7, 8]]), background=0)

        cache = SubtreeProgramCache()
        assert cache.lookup(g1, g2) is None

        prog = DSLProgram(instructions=[DSLInstruction("rotate_90", {})])
        cache.insert(g1, g2, prog)

        hit = cache.lookup(g1, g2)
        assert hit is not None
        assert hit.instructions[0].primitive == "rotate_90"
