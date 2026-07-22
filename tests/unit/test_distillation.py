"""
Unit tests for ProgramDistillation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY
from src.search.distillation import ProgramDistillation


class TestProgramDistillation:
    def test_distill_sequence(self):
        # Create a macro combining two rotations (should act as rotate_180)
        distiller = ProgramDistillation()
        distiller.distill_sequence(["rotate_90", "rotate_90"], "rotate_twice_macro")

        assert "rotate_twice_macro" in PRIMITIVE_REGISTRY
        g = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        macro_res = PRIMITIVE_REGISTRY["rotate_twice_macro"](g)
        expected_res = PRIMITIVE_REGISTRY["rotate_180"](g)

        assert (macro_res.pixels == expected_res.pixels).all()
