"""
LLMPromptGenerator — Formats ARC tasks into ASCII art, JSON, and s-expression prompts for LLM reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.grid.grid import ArcGrid


class LLMPromptGenerator:
    """Formats ARC tasks into structured prompts for LLM reasoning assistants."""

    @staticmethod
    def grid_to_ascii(grid: ArcGrid) -> str:
        """Converts ArcGrid to human-readable ASCII matrix representation."""
        lines = []
        for r in range(grid.height):
            line = " ".join(str(grid.get(r, c)) for c in range(grid.width))
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def generate_prompt(train_pairs: List[Dict[str, ArcGrid]], test_input: ArcGrid) -> str:
        """Generates a complete prompt with ASCII grids and instruction templates."""
        prompt_parts = ["You are an expert AI solving visual spatial reasoning tasks (ARC-AGI).\n"]
        prompt_parts.append("### Demonstrations:")

        for i, pair in enumerate(train_pairs, 1):
            prompt_parts.append(f"\nDemonstration {i}:")
            prompt_parts.append("INPUT:")
            prompt_parts.append(LLMPromptGenerator.grid_to_ascii(pair["input"]))
            prompt_parts.append("OUTPUT:")
            prompt_parts.append(LLMPromptGenerator.grid_to_ascii(pair["output"]))

        prompt_parts.append("\n### Test Input:")
        prompt_parts.append(LLMPromptGenerator.grid_to_ascii(test_input))
        prompt_parts.append("\nGenerate the output grid matrix for the test input.")

        return "\n".join(prompt_parts)
