"""
Unit tests for LLMValidator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.meta_learning.llm_validator import LLMValidator


class TestLLMValidator:
    def test_validate_candidate_pass(self):
        # Identity function matches
        fn = lambda g: g
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        score = LLMValidator.validate_candidate(fn, [(grid, grid)])
        assert score == 1.0

    def test_validate_candidate_fail(self):
        # Function returning different grid fails
        fn = lambda g: ArcGrid(pixels=np.zeros((2, 2), dtype=np.uint8), background=0)
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        score = LLMValidator.validate_candidate(fn, [(grid, grid)])
        assert score == 0.0
