"""
Unit tests for ProgramSlicing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.slicing import ProgramSlicing


class TestProgramSlicing:
    def test_slice_and_merge(self):
        # A grid containing colors 1 and 2
        g = ArcGrid(pixels=np.array([[1, 2], [0, 1]]), background=0)

        channels = ProgramSlicing.slice_channels(g)
        assert len(channels) == 2
        assert 1 in channels
        assert 2 in channels

        # Re-merge the channels to see if original grid is recovered
        merged = ProgramSlicing.merge_channels(list(channels.values()), background=0)
        assert (merged.pixels == g.pixels).all()
