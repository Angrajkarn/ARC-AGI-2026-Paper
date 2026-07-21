"""
Unit tests for LLMPromptGenerator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.llm.prompt_generator import LLMPromptGenerator


class TestPromptGenerator:
    def test_grid_to_ascii(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        ascii_str = LLMPromptGenerator.grid_to_ascii(grid)

        assert "1 2" in ascii_str
        assert "3 4" in ascii_str

    def test_generate_prompt(self):
        g1 = ArcGrid(pixels=np.array([[1, 1], [1, 1]]), background=0)
        g2 = ArcGrid(pixels=np.array([[2, 2], [2, 2]]), background=0)

        prompt = LLMPromptGenerator.generate_prompt([{"input": g1, "output": g2}], g1)
        assert "Demonstration 1" in prompt
        assert "Test Input" in prompt
