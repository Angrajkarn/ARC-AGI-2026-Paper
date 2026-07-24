"""
Unit tests for ProgramDistance.
"""

from __future__ import annotations

import pytest

from src.meta_learning.program_distance import ProgramDistance


class TestProgramDistance:
    def test_edit_distance_same(self):
        dist = ProgramDistance.edit_distance(["crop", "scale"], ["crop", "scale"])
        assert dist == 0

    def test_edit_distance_diff(self):
        dist = ProgramDistance.edit_distance(["crop", "scale"], ["crop", "rotate"])
        # One substitution: scale -> rotate
        assert dist == 1
