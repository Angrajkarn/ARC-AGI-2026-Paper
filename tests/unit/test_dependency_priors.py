"""
Unit tests for DependencyPriors.
"""

from __future__ import annotations

import pytest

from src.search.neural.dependency_priors import DependencyPriors


class TestDependencyPriors:
    def test_score_program_crop_scale(self):
        priors = DependencyPriors()
        score = priors.score_program(["crop_content", "scale_2x"])
        # crop -> scale is logically favored (1.2)
        assert score == 1.2

    def test_score_program_double_rotate(self):
        priors = DependencyPriors()
        score = priors.score_program(["rotate_90", "rotate_90"])
        # double rotate is penalized (0.1)
        assert score == 0.1
