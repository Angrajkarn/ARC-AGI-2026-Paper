"""
Unit tests for MultiModelLLMEnsemble.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.llm.llm_ensemble import LLMModelConfig, MultiModelLLMEnsemble


class TestLLMEnsemble:
    def test_ensemble_initialization(self):
        ensemble = MultiModelLLMEnsemble()
        assert len(ensemble.models) == 3

    def test_generate_candidate_programs(self):
        g1 = ArcGrid(pixels=np.array([[1, 1], [1, 1]]), background=0)
        ensemble = MultiModelLLMEnsemble()

        programs = ensemble.generate_candidate_programs([{"input": g1, "output": g1}], g1)
        assert len(programs) > 0
