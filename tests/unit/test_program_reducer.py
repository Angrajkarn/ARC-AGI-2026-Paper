"""
Unit tests for ProgramReducer.
"""

from __future__ import annotations

import pytest

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.meta_learning.program_reducer import ProgramReducer


class TestProgramReducer:
    def test_reduce_program_double_mirror(self):
        program = DSLProgram(
            instructions=[
                DSLInstruction(primitive="mirror_horizontal"),
                DSLInstruction(primitive="mirror_horizontal"),
                DSLInstruction(primitive="rotate_90"),
            ]
        )

        reduced = ProgramReducer.reduce_program(program)
        # Double mirror horizontal should cancel out, leaving just rotate_90
        assert len(reduced.instructions) == 1
        assert reduced.instructions[0].primitive == "rotate_90"
