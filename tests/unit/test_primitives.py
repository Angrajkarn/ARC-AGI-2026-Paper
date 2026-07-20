"""Unit tests for DSL primitives and executor."""

from __future__ import annotations

import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLInstruction, DSLParser, DSLProgram
from src.dsl.primitives.transforms import (
    PRIMITIVE_REGISTRY,
    add_border,
    crop,
    crop_to_content,
    flood_fill,
    gravity,
    mirror_horizontal,
    mirror_vertical,
    replace_color,
    rotate_90,
    scale,
    swap_colors,
    translate,
)


def make_grid(data):
    return ArcGrid.from_list(data)


class TestPrimitives:
    def test_rotate_90(self):
        g = make_grid([[1, 2], [3, 4]])
        r = rotate_90(g, times=1)
        assert r.to_list() == [[3, 1], [4, 2]]

    def test_rotate_180(self):
        g = make_grid([[1, 2], [3, 4]])
        r = rotate_90(g, times=2)
        assert r.to_list() == [[4, 3], [2, 1]]

    def test_rotate_360_identity(self):
        g = make_grid([[1, 2, 3], [4, 5, 6]])
        r = rotate_90(g, times=4)
        assert r == g

    def test_mirror_horizontal(self):
        g = make_grid([[1, 2, 3]])
        r = mirror_horizontal(g)
        assert r.to_list() == [[3, 2, 1]]

    def test_mirror_vertical(self):
        g = make_grid([[1, 2], [3, 4]])
        r = mirror_vertical(g)
        assert r.to_list() == [[3, 4], [1, 2]]

    def test_replace_color(self):
        g = make_grid([[1, 2, 1], [2, 1, 2]])
        r = replace_color(g, source_color=1, target_color=5)
        assert r.to_list() == [[5, 2, 5], [2, 5, 2]]

    def test_swap_colors(self):
        g = make_grid([[1, 2], [2, 1]])
        r = swap_colors(g, color_a=1, color_b=2)
        assert r.to_list() == [[2, 1], [1, 2]]

    def test_translate_down_right(self):
        g = make_grid([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        r = translate(g, dr=1, dc=1)
        assert r.get(1, 1) == 1

    def test_scale_2x(self):
        g = make_grid([[1, 2], [3, 4]])
        r = scale(g, factor=2)
        assert r.height == 4
        assert r.width == 4
        assert r.get(0, 0) == 1
        assert r.get(0, 1) == 1  # duplicated
        assert r.get(1, 0) == 1  # duplicated row
        assert r.get(2, 0) == 3

    def test_crop(self):
        g = make_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        r = crop(g, row_min=1, row_max=2, col_min=1, col_max=2)
        assert r.to_list() == [[5, 6], [8, 9]]

    def test_crop_to_content(self):
        g = make_grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = crop_to_content(g)
        assert r.to_list() == [[1]]

    def test_flood_fill(self):
        g = make_grid([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        r = flood_fill(g, row=1, col=1, new_color=5)
        assert r.get(1, 1) == 5
        assert r.get(0, 0) == 1  # unchanged

    def test_add_border(self):
        g = make_grid([[1]])
        r = add_border(g, color=2, thickness=1)
        assert r.height == 3
        assert r.width == 3
        assert r.get(0, 0) == 2
        assert r.get(1, 1) == 1

    def test_gravity_down(self):
        g = make_grid([[1, 0], [0, 0]])
        r = gravity(g, direction="down")
        assert r.get(1, 0) == 1
        assert r.get(0, 0) == 0

    def test_gravity_right(self):
        g = make_grid([[1, 0, 0]])
        r = gravity(g, direction="right")
        assert r.get(0, 2) == 1

    def test_registry_populated(self):
        assert "rotate_90" in PRIMITIVE_REGISTRY
        assert "mirror_horizontal" in PRIMITIVE_REGISTRY
        assert "replace_color" in PRIMITIVE_REGISTRY
        assert len(PRIMITIVE_REGISTRY) >= 25


class TestDSLParser:
    def test_from_names(self):
        prog = DSLParser.from_names(["rotate_90", "mirror_horizontal"])
        assert prog.length == 2
        assert prog.instructions[0].primitive == "rotate_90"

    def test_from_op_list(self):
        ops = [
            {"op": "rotate_90", "args": {"times": 2}},
            {"op": "replace_color", "args": {"source_color": 1, "target_color": 3}},
        ]
        prog = DSLParser.from_op_list(ops)
        assert prog.instructions[1].args["source_color"] == 1

    def test_roundtrip_json(self):
        prog = DSLParser.from_names(["mirror_horizontal", "rotate_90"])
        prog2 = DSLProgram.from_json(prog.to_json())
        assert prog.length == prog2.length

    def test_validate_invalid_primitive(self):
        prog = DSLParser.from_names(["not_a_real_primitive"])
        errors = DSLParser.validate(prog)
        assert len(errors) > 0


class TestDSLExecutor:
    def test_execute_identity(self):
        executor = DSLExecutor()
        g = make_grid([[1, 2], [3, 4]])
        prog = DSLProgram(instructions=[], name="identity")
        result, trace = executor.execute(prog, g)
        assert result == g
        assert trace.success

    def test_execute_rotate(self):
        executor = DSLExecutor()
        g = make_grid([[1, 2], [3, 4]])
        prog = DSLParser.from_op_list([{"op": "rotate_90", "args": {"times": 1}}])
        result, trace = executor.execute(prog, g)
        assert result is not None
        assert result.to_list() == [[3, 1], [4, 2]]
        assert trace.success

    def test_execute_chain(self):
        executor = DSLExecutor()
        g = make_grid([[1, 2], [3, 4]])
        prog = DSLParser.from_names(["rotate_90", "mirror_horizontal"])
        result, trace = executor.execute(prog, g)
        assert result is not None
        assert len(trace.steps) == 2

    def test_execute_invalid_primitive(self):
        executor = DSLExecutor()
        g = make_grid([[1, 2]])
        instr = DSLInstruction(primitive="nonexistent_primitive")
        prog = DSLProgram(instructions=[instr])
        result, trace = executor.execute(prog, g)
        assert not trace.success

    def test_input_not_mutated(self):
        import numpy as np
        executor = DSLExecutor()
        data = [[1, 2], [3, 4]]
        g = make_grid(data)
        prog = DSLParser.from_names(["replace_color"])
        # Even with a modifying operation, original should not change
        result, _ = executor.execute(
            DSLParser.from_op_list([{"op": "replace_color", "args": {"source_color": 1, "target_color": 9}}]),
            g,
        )
        assert g.to_list() == data  # original unchanged
