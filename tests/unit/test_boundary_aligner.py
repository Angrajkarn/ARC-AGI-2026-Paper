"""
Unit tests for BoundaryAligner.
"""

from __future__ import annotations

import pytest

from src.core.graphs.boundary_aligner import BoundaryAligner


class TestBoundaryAligner:
    def test_check_alignment_true(self):
        obj1 = {(0, 0), (0, 1)}
        obj2 = {(0, 3), (1, 3)}

        h_align, v_align = BoundaryAligner.check_alignment(obj1, obj2)

        # Share row 0 -> h_align is True, no column overlap -> v_align is False
        assert h_align
        assert not v_align

    def test_check_alignment_none(self):
        obj1 = {(0, 0)}
        obj2 = {(2, 2)}

        h_align, v_align = BoundaryAligner.check_alignment(obj1, obj2)
        assert not h_align
        assert not v_align
