"""
Unit tests for ASTGeneticOperators.
"""

from __future__ import annotations

import pytest

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.search.genetic.genetic_operators import ASTGeneticOperators


class TestASTGeneticOperators:
    def test_crossover(self):
        p1 = DSLProgram(instructions=[
            DSLInstruction("rotate_90", {}),
            DSLInstruction("mirror_horizontal", {}),
        ])
        p2 = DSLProgram(instructions=[
            DSLInstruction("crop_to_content", {}),
            DSLInstruction("fill_holes", {}),
        ])

        c1, c2 = ASTGeneticOperators.crossover(p1, p2)
        assert isinstance(c1, DSLProgram)
        assert isinstance(c2, DSLProgram)

    def test_mutate(self):
        p = DSLProgram(instructions=[
            DSLInstruction("rotate_90", {"times": 1}),
        ])

        mutated = ASTGeneticOperators.mutate(p, mutation_rate=1.0)
        assert isinstance(mutated, DSLProgram)
