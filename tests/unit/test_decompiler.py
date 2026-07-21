"""
Unit tests for ASTDecompiler.
"""

from __future__ import annotations

import pytest

from src.dsl.decompiler.decompiler import ASTDecompiler
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram


class TestDecompiler:
    def test_to_natural_language(self):
        prog = DSLProgram(instructions=[
            DSLInstruction("rotate_90", {"times": 1}),
            DSLInstruction("fill_holes", {}),
        ])

        text = ASTDecompiler.to_natural_language(prog)
        assert "1. Apply primitive 'rotate_90'" in text
        assert "2. Apply primitive 'fill_holes'" in text

    def test_to_latex(self):
        prog = DSLProgram(instructions=[
            DSLInstruction("rotate_90", {}),
        ])

        latex = ASTDecompiler.to_latex(prog)
        assert r"G_{\text{out}}" in latex
        assert r"\text{rotate_90}" in latex
