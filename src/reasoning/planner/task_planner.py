"""
Task Planner — high-level planning for ARC task solving.

Given a task (training pairs + test inputs), the planner:
  1. Extracts features from all training pairs.
  2. Assesses task difficulty.
  3. Selects the best search strategy.
  4. Allocates computational budget.
  5. Returns a TaskPlan used by the SearchEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis, RuleDiscoverer
from src.vision.features.feature_extractor import FeatureExtractor, FeatureVector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TaskPlan:
    """Execution plan produced by the TaskPlanner.

    Attributes:
        search_algorithm: Algorithm to use (beam | mcts | genetic | constraint).
        time_budget_sec:  Maximum solving time in seconds.
        beam_width:       Beam width (if beam search selected).
        max_iterations:   Maximum search iterations.
        hypotheses:       Ranked list of candidate hypotheses.
        features_in:      Feature vectors for training inputs.
        features_out:     Feature vectors for training outputs.
        difficulty:       Estimated difficulty (low | medium | high).
        notes:            Planning notes for logging/debugging.
    """

    search_algorithm: str = "beam"
    time_budget_sec: float = 60.0
    beam_width: int = 10
    max_iterations: int = 500
    hypotheses: List[Hypothesis] = field(default_factory=list)
    features_in: List[FeatureVector] = field(default_factory=list)
    features_out: List[FeatureVector] = field(default_factory=list)
    shape_predictions: List[Any] = field(default_factory=list)
    difficulty: str = "medium"
    notes: List[str] = field(default_factory=list)


class TaskPlanner:
    """Plan how to solve an ARC task.

    Usage::

        planner = TaskPlanner(config)
        plan = planner.plan(task)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        from src.reasoning.planner.shape_predictor import ShapePredictor
        self._extractor = FeatureExtractor()
        self._discoverer = RuleDiscoverer(top_k=20)
        self._shape_predictor = ShapePredictor()
        self._config = config

    def plan(self, task: Dict) -> TaskPlan:
        """Generate a TaskPlan for the given ARC task.

        Args:
            task: ARC task dict with "train" pairs (each having "input"/"output" ArcGrids).

        Returns:
            TaskPlan instance.
        """
        pairs = task.get("train", [])
        plan = TaskPlan()

        if not pairs:
            plan.notes.append("No training pairs — returning default plan")
            return plan

        # Feature extraction
        plan.features_in = [self._extractor.extract(p["input"]) for p in pairs]
        plan.features_out = [self._extractor.extract(p["output"]) for p in pairs]

        # Shape prediction
        test_pairs = task.get("test", [])
        if test_pairs and "input" in test_pairs[0]:
            plan.shape_predictions = self._shape_predictor.predict(pairs, test_pairs[0]["input"])

        # Rule discovery
        plan.hypotheses = self._discoverer.discover(pairs)
        logger.info(f"Planning: {len(plan.hypotheses)} hypotheses found for {len(pairs)} pairs")

        # Difficulty assessment
        plan.difficulty = self._assess_difficulty(pairs, plan.features_in, plan.features_out)

        # Algorithm selection
        plan.search_algorithm = self._select_algorithm(plan)

        # Budget allocation
        plan.time_budget_sec = self._allocate_budget(plan)
        plan.beam_width = self._allocate_beam_width(plan)
        plan.max_iterations = self._allocate_iterations(plan)

        plan.notes.append(
            f"Difficulty={plan.difficulty}, algorithm={plan.search_algorithm}, "
            f"budget={plan.time_budget_sec}s, beam_width={plan.beam_width}"
        )

        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_difficulty(
        self,
        pairs: List[Dict],
        features_in: List[FeatureVector],
        features_out: List[FeatureVector],
    ) -> str:
        """Heuristic difficulty: low / medium / high."""
        n_colors = max((f.color.num_colors for f in features_in), default=0)
        n_objects = max((f.objects.num_objects for f in features_in), default=0)

        # Size change between input and output
        size_changes = [
            p["input"].size != p["output"].size for p in pairs
        ]
        any_size_change = any(size_changes)

        # Score
        score = 0
        if n_colors > 3:
            score += 1
        if n_objects > 5:
            score += 1
        if any_size_change:
            score += 1
        if len(pairs) > 4:
            score += 1

        if score <= 1:
            return "low"
        if score <= 2:
            return "medium"
        return "high"

    def _select_algorithm(self, plan: TaskPlan) -> str:
        """Choose the best search algorithm based on difficulty and hypotheses."""
        # If a hypothesis already has conf=1.0 → pure beam is fastest
        top_conf = plan.hypotheses[0].confidence if plan.hypotheses else 0.0
        if top_conf == 1.0:
            return "beam"

        if plan.difficulty == "low":
            return "beam"
        if plan.difficulty == "medium":
            return "beam"
        # High difficulty: try MCTS for deeper search
        return "mcts"

    def _allocate_budget(self, plan: TaskPlan) -> float:
        base = 30.0
        if plan.difficulty == "low":
            return base
        if plan.difficulty == "medium":
            return base * 2
        return base * 4  # 120s for hard tasks

    def _allocate_beam_width(self, plan: TaskPlan) -> int:
        if plan.difficulty == "low":
            return 5
        if plan.difficulty == "medium":
            return 10
        return 20

    def _allocate_iterations(self, plan: TaskPlan) -> int:
        if plan.difficulty == "low":
            return 100
        if plan.difficulty == "medium":
            return 300
        return 800


__all__ = ["TaskPlanner", "TaskPlan"]
