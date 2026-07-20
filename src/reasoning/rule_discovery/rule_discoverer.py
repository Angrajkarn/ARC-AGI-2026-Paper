"""
Rule Discoverer — infer transformation rules from ARC training pairs.

Given a list of (input_grid, output_grid) training pairs, this module
generates ranked Hypothesis objects describing the transformation.

Strategy:
  1. Extract features from each input and output.
  2. Compare features to identify changes.
  3. Match changes against a library of rule templates.
  4. Score hypotheses by how many pairs they explain.
  5. Return ranked list.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.vision.features.feature_extractor import FeatureExtractor, FeatureVector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A candidate transformation rule.

    Attributes:
        description:     Human-readable description.
        confidence:      Score in [0, 1], higher = more evidence.
        candidate_program: DSLProgram that implements the rule.
        evidence_count:  Number of training pairs this hypothesis explains.
        rule_type:       Category of rule (color, rotation, object, etc.)
        params:          Extra parameters discovered during induction.
    """

    description: str
    confidence: float
    candidate_program: DSLProgram
    evidence_count: int = 0
    rule_type: str = "unknown"
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Hypothesis(type={self.rule_type!r}, conf={self.confidence:.2f}, "
            f"desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Rule templates
# ---------------------------------------------------------------------------

def _make_program(ops: List[Dict]) -> DSLProgram:
    instructions = [DSLInstruction(primitive=op["op"], args=op.get("args", {})) for op in ops]
    return DSLProgram(instructions=instructions)


class RuleDiscoverer:
    """Discover transformation rules from ARC training pairs.

    Usage::

        discoverer = RuleDiscoverer()
        pairs = [{"input": grid_in, "output": grid_out}, ...]
        hypotheses = discoverer.discover(pairs)
    """

    def __init__(self, top_k: int = 15) -> None:
        from src.reasoning.color_rules import ColorRulesSolver
        from src.reasoning.object_rules import ObjectRulesEngine
        from src.reasoning.rule_templates import RuleTemplateSynthesizer

        self._extractor = FeatureExtractor()
        self._detector = ObjectDetector()
        self._color_solver = ColorRulesSolver()
        self._object_engine = ObjectRulesEngine()
        self._template_synthesizer = RuleTemplateSynthesizer()
        self.top_k = top_k

    def discover(self, pairs: List[Dict]) -> List[Hypothesis]:
        """Generate ranked hypotheses from training pairs.

        Args:
            pairs: List of dicts with "input" and "output" ArcGrid values.

        Returns:
            List of Hypothesis objects, sorted by confidence descending.
        """
        if not pairs:
            return []

        input_features = [self._extractor.extract(p["input"]) for p in pairs]
        output_features = [self._extractor.extract(p["output"]) for p in pairs]

        hypotheses: List[Hypothesis] = []

        # 1. Advanced Engine Rule Discovery (Color Matrix, Object Rules, Template Pipelines)
        color_results = self._color_solver.solve(pairs)
        for c_res in color_results:
            progs = self._color_solver.build_dsl_programs(c_res)
            for prog in progs:
                hypotheses.append(
                    Hypothesis(
                        description=f"Color rule ({c_res.rule_type})",
                        confidence=c_res.confidence,
                        candidate_program=prog,
                        evidence_count=len(pairs) if c_res.confidence == 1.0 else 1,
                        rule_type="color",
                    )
                )

        obj_rules = self._object_engine.discover_object_rules(pairs)
        for o_rule in obj_rules:
            hypotheses.append(
                Hypothesis(
                    description=o_rule.description,
                    confidence=o_rule.confidence,
                    candidate_program=o_rule.program,
                    evidence_count=len(pairs) if o_rule.confidence == 1.0 else 1,
                    rule_type=o_rule.rule_type,
                    params=o_rule.params,
                )
            )

        template_hyps = self._template_synthesizer.synthesize_templates(pairs)
        for t_hyp in template_hyps:
            hypotheses.append(
                Hypothesis(
                    description=t_hyp.description,
                    confidence=t_hyp.confidence,
                    candidate_program=t_hyp.program,
                    evidence_count=len(pairs) if t_hyp.confidence == 1.0 else 1,
                    rule_type="composite_template",
                )
            )

        # 2. Atomic Rule Checks
        hypotheses.extend(self._check_no_change(pairs))
        hypotheses.extend(self._check_color_replacement(pairs, input_features, output_features))
        hypotheses.extend(self._check_rotation(pairs))
        hypotheses.extend(self._check_mirror(pairs))
        hypotheses.extend(self._check_gravity(pairs))
        hypotheses.extend(self._check_size_change(pairs, input_features, output_features))
        hypotheses.extend(self._check_border_added(pairs, input_features, output_features))
        hypotheses.extend(self._check_fill_holes(pairs))
        hypotheses.extend(self._check_crop_to_content(pairs))
        hypotheses.extend(self._check_color_swap(pairs))
        hypotheses.extend(self._check_outline(pairs, input_features, output_features))
        hypotheses.extend(self._check_hollow(pairs))
        hypotheses.extend(self._check_tiling(pairs, input_features, output_features))
        hypotheses.extend(self._check_scale(pairs, input_features, output_features))

        # Deduplicate by description
        seen: Set[str] = set()
        unique: List[Hypothesis] = []
        for h in hypotheses:
            if h.description not in seen:
                seen.add(h.description)
                unique.append(h)

        # If no exact hypotheses found or fewer than top_k, add feature-guided seed candidates
        if len(unique) < self.top_k:
            feature_seeds = self._generate_feature_seeds(pairs, input_features, output_features)
            for h in feature_seeds:
                if h.description not in seen:
                    seen.add(h.description)
                    unique.append(h)

        # Sort by confidence then evidence count
        unique.sort(key=lambda h: (h.confidence, h.evidence_count), reverse=True)

        logger.debug(f"Discovered {len(unique)} hypotheses from {len(pairs)} pairs")
        return unique[: self.top_k]

    def _generate_feature_seeds(
        self, pairs: List[Dict], in_feats: List[FeatureVector], out_feats: List[FeatureVector]
    ) -> List[Hypothesis]:
        """Generate heuristic seed hypotheses based on input/output feature differences."""
        seeds: List[Hypothesis] = []
        if not pairs:
            return seeds

        in_colors = set().union(*(f.color.color_set for f in in_feats))
        out_colors = set().union(*(f.color.color_set for f in out_feats))
        new_colors = out_colors - in_colors

        # 1. Color replacement seeds for new output colors
        for src in in_colors:
            for tgt in out_colors:
                if src != tgt:
                    prog = _make_program([
                        {"op": "replace_color", "args": {"source_color": src, "target_color": tgt}}
                    ])
                    correct, conf = self._verify_program(prog, pairs)
                    seeds.append(Hypothesis(
                        description=f"Seed replace {src} -> {tgt}",
                        confidence=max(conf, 0.1 if tgt in new_colors else 0.05),
                        candidate_program=prog,
                        evidence_count=correct,
                        rule_type="color",
                    ))

        # 2. Geometric transform seeds
        for op, desc in [
            ("rotate_90", "Seed rotate 90"),
            ("mirror_horizontal", "Seed mirror H"),
            ("mirror_vertical", "Seed mirror V"),
            ("crop_to_content", "Seed crop to content"),
            ("fill_holes", "Seed fill holes"),
            ("hollow", "Seed hollow"),
            ("normalize_colors", "Seed normalize colors"),
        ]:
            prog = _make_program([{"op": op}])
            correct, conf = self._verify_program(prog, pairs)
            seeds.append(Hypothesis(
                description=desc,
                confidence=max(conf, 0.05),
                candidate_program=prog,
                evidence_count=correct,
                rule_type="geometry",
            ))

        # 3. Gravity seeds
        for d in ["down", "up", "left", "right"]:
            prog = _make_program([{"op": "gravity", "args": {"direction": d}}])
            correct, conf = self._verify_program(prog, pairs)
            seeds.append(Hypothesis(
                description=f"Seed gravity {d}",
                confidence=max(conf, 0.05),
                candidate_program=prog,
                evidence_count=correct,
                rule_type="gravity",
            ))

        seeds.sort(key=lambda h: h.confidence, reverse=True)
        return seeds

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def _verify_program(self, program: DSLProgram, pairs: List[Dict]) -> Tuple[int, float]:
        """Verify a program against all pairs. Returns (count_correct, confidence)."""
        from src.dsl.executor.executor import DSLExecutor
        executor = DSLExecutor(debug=False)
        correct = 0
        for pair in pairs:
            result, trace = executor.execute(program, pair["input"])
            if result is not None and result == pair["output"]:
                correct += 1
        confidence = correct / max(len(pairs), 1)
        return correct, confidence

    def _check_no_change(self, pairs: List[Dict]) -> List[Hypothesis]:
        program = DSLProgram(instructions=[], name="identity")
        correct = sum(1 for p in pairs if p["input"] == p["output"])
        conf = correct / max(len(pairs), 1)
        if conf > 0:
            return [Hypothesis(
                description="Identity (no change)",
                confidence=conf,
                candidate_program=program,
                evidence_count=correct,
                rule_type="identity",
            )]
        return []

    def _check_color_replacement(
        self, pairs, in_feats, out_feats
    ) -> List[Hypothesis]:
        results = []
        # Find consistent color mappings across all pairs
        all_pairs_mappings: List[Optional[Dict[int, int]]] = []
        for pair in pairs:
            mapping = self._infer_color_mapping(pair["input"], pair["output"])
            all_pairs_mappings.append(mapping)

        # Find mappings consistent across all pairs
        if all(m is not None for m in all_pairs_mappings):
            # Check if all mappings agree
            first = all_pairs_mappings[0]
            if all(m == first for m in all_pairs_mappings):
                for src, tgt in first.items():
                    if src != tgt:
                        prog = _make_program([
                            {"op": "replace_color", "args": {"source_color": src, "target_color": tgt}}
                        ])
                        correct, conf = self._verify_program(prog, pairs)
                        if conf > 0:
                            results.append(Hypothesis(
                                description=f"Replace colour {src} -> {tgt}",
                                confidence=conf,
                                candidate_program=prog,
                                evidence_count=correct,
                                rule_type="color",
                                params={"source": src, "target": tgt},
                            ))
        return results

    def _infer_color_mapping(
        self, grid_in: ArcGrid, grid_out: ArcGrid
    ) -> Optional[Dict[int, int]]:
        """Infer a consistent pixel-level colour mapping (if any)."""
        if grid_in.size != grid_out.size:
            return None
        mapping: Dict[int, int] = {}
        for src, tgt in zip(grid_in.pixels.flatten(), grid_out.pixels.flatten()):
            src, tgt = int(src), int(tgt)
            if src in mapping and mapping[src] != tgt:
                return None
            mapping[src] = tgt
        return mapping

    def _check_rotation(self, pairs: List[Dict]) -> List[Hypothesis]:
        results = []
        for times, name in [(1, "90°"), (2, "180°"), (3, "270°")]:
            prog = _make_program([{"op": "rotate_90", "args": {"times": times}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Rotate {name} clockwise",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="rotation",
                ))
        return results

    def _check_mirror(self, pairs: List[Dict]) -> List[Hypothesis]:
        results = []
        for op, desc in [
            ("mirror_horizontal", "Mirror left-right"),
            ("mirror_vertical", "Mirror top-bottom"),
            ("mirror_diagonal", "Mirror diagonal"),
        ]:
            prog = _make_program([{"op": op}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=desc,
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="reflection",
                ))
        return results

    def _check_gravity(self, pairs: List[Dict]) -> List[Hypothesis]:
        results = []
        for direction in ["down", "up", "left", "right"]:
            prog = _make_program([{"op": "gravity", "args": {"direction": direction}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Gravity ({direction})",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="gravity",
                ))
        return results

    def _check_size_change(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        results = []
        # Check if output = input scaled by integer factor
        for pair, inf, outf in zip(pairs, in_feats, out_feats):
            ih, iw = pair["input"].size
            oh, ow = pair["output"].size
            if oh == 0 or ow == 0:
                continue
            if oh % ih == 0 and ow % iw == 0:
                fh, fw = oh // ih, ow // iw
                if fh == fw and 2 <= fh <= 5:
                    prog = _make_program([{"op": "scale", "args": {"factor": fh}}])
                    correct, conf = self._verify_program(prog, pairs)
                    if conf > 0:
                        results.append(Hypothesis(
                            description=f"Scale ×{fh}",
                            confidence=conf,
                            candidate_program=prog,
                            evidence_count=correct,
                            rule_type="scale",
                        ))
                    break
        return results

    def _check_crop_to_content(self, pairs: List[Dict]) -> List[Hypothesis]:
        prog = _make_program([{"op": "crop_to_content"}])
        correct, conf = self._verify_program(prog, pairs)
        if conf > 0:
            return [Hypothesis(
                description="Crop to content bounding box",
                confidence=conf,
                candidate_program=prog,
                evidence_count=correct,
                rule_type="crop",
            )]
        return []

    def _check_border_added(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        results = []
        for color in range(10):
            prog = _make_program([{"op": "add_border", "args": {"color": color, "thickness": 1}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Add border (colour {color})",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="border",
                ))
        return results

    def _check_fill_holes(self, pairs: List[Dict]) -> List[Hypothesis]:
        prog = _make_program([{"op": "fill_holes"}])
        correct, conf = self._verify_program(prog, pairs)
        if conf > 0:
            return [Hypothesis(
                description="Fill enclosed holes",
                confidence=conf,
                candidate_program=prog,
                evidence_count=correct,
                rule_type="fill",
            )]
        return []

    def _check_color_swap(self, pairs: List[Dict]) -> List[Hypothesis]:
        results = []
        colors = list(range(10))
        for a, b in itertools.combinations(colors, 2):
            prog = _make_program([{"op": "swap_colors", "args": {"color_a": a, "color_b": b}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Swap colours {a}↔{b}",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="color_swap",
                ))
        return results

    def _check_outline(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        results = []
        for color in range(10):
            prog = _make_program([{"op": "outline", "args": {"color": color}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Outline objects (colour {color})",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="outline",
                ))
        return results

    def _check_hollow(self, pairs: List[Dict]) -> List[Hypothesis]:
        prog = _make_program([{"op": "hollow"}])
        correct, conf = self._verify_program(prog, pairs)
        if conf > 0:
            return [Hypothesis(
                description="Make objects hollow (border only)",
                confidence=conf,
                candidate_program=prog,
                evidence_count=correct,
                rule_type="hollow",
            )]
        return []

    def _check_tiling(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        results = []
        for r in range(1, 4):
            for c in range(1, 4):
                if r == 1 and c == 1:
                    continue
                prog = _make_program([{"op": "tile", "args": {"rows": r, "cols": c}}])
                correct, conf = self._verify_program(prog, pairs)
                if conf > 0:
                    results.append(Hypothesis(
                        description=f"Tile {r}×{c}",
                        confidence=conf,
                        candidate_program=prog,
                        evidence_count=correct,
                        rule_type="tile",
                    ))
        return results

    def _check_scale(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        results = []
        for factor in [2, 3, 4]:
            prog = _make_program([{"op": "scale", "args": {"factor": factor}}])
            correct, conf = self._verify_program(prog, pairs)
            if conf > 0:
                results.append(Hypothesis(
                    description=f"Scale ×{factor}",
                    confidence=conf,
                    candidate_program=prog,
                    evidence_count=correct,
                    rule_type="scale",
                ))
        return results

    def _check_object_count_change(self, pairs, in_feats, out_feats) -> List[Hypothesis]:
        """Placeholder: detect cases where the number of objects changes."""
        return []


__all__ = ["RuleDiscoverer", "Hypothesis"]
