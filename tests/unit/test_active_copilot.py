"""
Unit tests for ActiveCopilot.
"""

from __future__ import annotations

import pytest

from src.meta_learning.active_copilot import ActiveCopilot


class TestActiveCopilot:
    def test_active_feedback_boost(self):
        copilot = ActiveCopilot()
        copilot.register_feedback("rotate")

        candidates = ["scale_2x", "rotate_90", "mirror_vertical"]
        priors = copilot.rerank_priors(candidates)

        # rotate_90 should receive priority boost (2.5 vs 1.0)
        assert priors["rotate_90"] == 2.5
        assert priors["scale_2x"] == 1.0
