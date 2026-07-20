"""Unit tests for Phase 2 extensions: ColorRulesSolver, ObjectRulesEngine, AST nodes, and ParallelEvaluator."""

from __future__ import annotations

import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.ast.ast_nodes import ApplyWithMaskNode, PrimitiveNode, SequenceNode
from src.reasoning.color_rules import ColorRulesSolver
from src.reasoning.object_rules import ObjectRulesEngine


def make_grid(data, bg=0):
    return ArcGrid.from_list(data, background=bg)


class TestColorRulesSolver:
    def test_direct_color_mapping(self):
        pairs = [
            {"input": make_grid([[1, 2], [2, 1]]), "output": make_grid([[3, 4], [4, 3]])},
            {"input": make_grid([[2, 1]]), "output": make_grid([[4, 3]])},
        ]
        solver = ColorRulesSolver()
        results = solver.solve(pairs)
        assert len(results) >= 1
        top = results[0]
        assert top.mapping[1] == 3
        assert top.mapping[2] == 4
        assert top.confidence == 1.0

    def test_frequency_rank_color_mapping(self):
        pairs = [
            # In: 1 appears 3 times, 2 appears 1 time -> Out: 5 appears 3 times, 6 appears 1 time
            {"input": make_grid([[1, 1], [1, 2]]), "output": make_grid([[5, 5], [5, 6]])},
        ]
        solver = ColorRulesSolver()
        results = solver.solve(pairs)
        assert any(r.rule_type == "frequency_rank" for r in results)


class TestObjectRulesEngine:
    def test_discover_isolate_largest(self):
        pairs = [
            {"input": make_grid([[1, 1, 0, 2], [1, 1, 0, 0]], bg=0),
             "output": make_grid([[1, 1, 0, 0], [1, 1, 0, 0]], bg=0)},
        ]
        engine = ObjectRulesEngine()
        rules = engine.discover_object_rules(pairs)
        assert any(r.rule_type == "isolate_object" and r.params.get("mode") == "largest" for r in rules)

    def test_discover_isolate_by_color(self):
        pairs = [
            {"input": make_grid([[3, 3, 0, 2], [3, 3, 0, 0]], bg=0),
             "output": make_grid([[3, 3, 0, 0], [3, 3, 0, 0]], bg=0)},
        ]
        engine = ObjectRulesEngine()
        rules = engine.discover_object_rules(pairs)
        assert any(r.rule_type == "isolate_by_color" and r.params.get("color") == 3 for r in rules)


class TestASTNodes:
    def test_primitive_node(self):
        node = PrimitiveNode(primitive="rotate_90", args={"times": 1})
        grid = make_grid([[1, 2], [3, 4]])
        res = node.evaluate(grid)
        assert res.to_list() == [[3, 1], [4, 2]]

    def test_sequence_node(self):
        n1 = PrimitiveNode(primitive="rotate_90", args={"times": 1})
        n2 = PrimitiveNode(primitive="mirror_horizontal")
        seq = SequenceNode(children=[n1, n2])
        grid = make_grid([[1, 2], [3, 4]])
        res = seq.evaluate(grid)
        assert res is not None

    def test_apply_with_mask_node(self):
        # Body transforms color 1 to 5, but mask only applies to pixels where grid == 1
        body = PrimitiveNode(primitive="replace_color", args={"source_color": 1, "target_color": 5})
        node = ApplyWithMaskNode(body_node=body, mask_color=1)
        grid = make_grid([[1, 2], [1, 3]])
        res = node.evaluate(grid)
        assert res.get(0, 0) == 5
        assert res.get(0, 1) == 2  # color 2 untouched
