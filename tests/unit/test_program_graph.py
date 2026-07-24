"""
Unit tests for ProgramGraphEmbedder.
"""

from __future__ import annotations

import pytest

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.core.graphs.program_graph import ProgramGraphEmbedder


class TestProgramGraphEmbedder:
    def test_build_program_graph(self):
        program = DSLProgram(
            instructions=[
                DSLInstruction(primitive="crop_content"),
                DSLInstruction(primitive="scale_2x"),
            ]
        )

        p_graph = ProgramGraphEmbedder.build_program_graph(program)

        # 2 nodes and 1 dependency edge
        assert len(p_graph["nodes"]) == 2
        assert p_graph["nodes"][0] == "crop_content"
        assert p_graph["nodes"][1] == "scale_2x"
        assert p_graph["edges"] == [(0, 1)]
