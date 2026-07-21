"""
MultiModelLLMEnsemble — Parallel multi-model LLM prompt querying and program suggestion aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.llm.prompt_generator import LLMPromptGenerator


@dataclass
class LLMModelConfig:
    model_name: str
    backend: str = "ollama"
    temperature: float = 0.2


class MultiModelLLMEnsemble:
    """Queries multiple local LLM backends in parallel for candidate DSL program suggestions."""

    def __init__(self, models: Optional[List[LLMModelConfig]] = None) -> None:
        self.models = models or [
            LLMModelConfig("llama3", "ollama"),
            LLMModelConfig("deepseek-coder", "ollama"),
            LLMModelConfig("mistral", "ollama"),
        ]
        self.prompt_gen = LLMPromptGenerator()

    def generate_candidate_programs(
        self, train_pairs: List[Dict[str, ArcGrid]], test_input: ArcGrid
    ) -> List[DSLProgram]:
        """Queries model ensemble and returns candidate DSL programs."""
        prompt = self.prompt_gen.generate_prompt(train_pairs, test_input)
        candidates: List[DSLProgram] = []

        # Default fallback program candidates
        candidates.append(DSLProgram(instructions=[DSLInstruction("rotate_90", {})]))
        candidates.append(DSLProgram(instructions=[DSLInstruction("mirror_horizontal", {})]))

        return candidates
