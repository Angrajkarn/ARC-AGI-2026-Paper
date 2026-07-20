"""
Monte Carlo Tree Search (MCTS) for ARC DSL program synthesis.

Implements UCB1-guided tree search over the program space.
Each node represents a partial DSL program.
Rollouts execute the program and score it on training pairs.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Primitives available during MCTS expansion (subset for speed)
_MCTS_PRIMITIVES = [
    ("mirror_horizontal", {}),
    ("mirror_vertical", {}),
    ("rotate_90", {"times": 1}),
    ("rotate_90", {"times": 2}),
    ("rotate_90", {"times": 3}),
    ("crop_to_content", {}),
    ("fill_holes", {}),
    ("gravity", {"direction": "down"}),
    ("gravity", {"direction": "up"}),
    ("gravity", {"direction": "left"}),
    ("gravity", {"direction": "right"}),
    ("normalize_colors", {}),
    ("outline", {"color": 0}),
    ("hollow", {}),
    ("scale", {"factor": 2}),
] + [
    ("replace_color", {"source_color": s, "target_color": t})
    for s in range(10) for t in range(10) if s != t
]


@dataclass
class MCTSNode:
    """A node in the MCTS tree representing a partial program."""

    program: DSLProgram
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    def ucb1(self, exploration_c: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        return self.avg_reward + exploration_c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class MCTSResult:
    best_program: Optional[DSLProgram]
    best_score: float
    iterations: int
    elapsed_sec: float
    found_perfect: bool


class MCTS:
    """Monte Carlo Tree Search for ARC DSL synthesis.

    Usage::

        mcts = MCTS(simulations=200, max_depth=8)
        result = mcts.search(hypotheses, pairs, time_budget=60)
    """

    def __init__(
        self,
        simulations: int = 200,
        exploration_constant: float = 1.414,
        max_depth: int = 10,
        rollout_depth: int = 3,
        seed: int = 42,
    ) -> None:
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self._executor = DSLExecutor(debug=False)
        self._rng = random.Random(seed)

    def search(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 60.0,
    ) -> MCTSResult:
        """Run MCTS search.

        Args:
            hypotheses: Initial hypotheses (used to init root children).
            pairs:      Training pairs.
            time_budget: Wall-clock time limit.

        Returns:
            MCTSResult with best program.
        """
        t_start = time.perf_counter()

        # Create root (empty program)
        root = MCTSNode(program=DSLProgram(instructions=[], source="mcts"))
        best_program: Optional[DSLProgram] = None
        best_score = 0.0

        # Seed root children from hypotheses
        for h in hypotheses[:5]:
            child = MCTSNode(
                program=h.candidate_program,
                parent=root,
                depth=h.candidate_program.length,
            )
            root.children.append(child)
            score = self._evaluate(child.program, pairs)
            child.visits = 1
            child.total_reward = score
            root.visits += 1
            root.total_reward += score
            if score > best_score:
                best_score = score
                best_program = child.program
            if score == 1.0:
                elapsed = time.perf_counter() - t_start
                return MCTSResult(
                    best_program=best_program,
                    best_score=best_score,
                    iterations=1,
                    elapsed_sec=elapsed,
                    found_perfect=True,
                )

        for iteration in range(self.simulations):
            if time.perf_counter() - t_start > time_budget:
                break

            # Selection
            node = self._select(root)

            # Expansion
            if node.depth < self.max_depth and not node.is_leaf():
                node = self._expand(node)
            elif node.depth < self.max_depth:
                node = self._expand(node)

            # Simulation (rollout)
            score = self._rollout(node, pairs)

            # Back-propagation
            self._backpropagate(node, score)

            if score > best_score:
                best_score = score
                best_program = node.program
                logger.debug(
                    f"MCTS iter {iteration}: new best score={score:.3f}, "
                    f"depth={node.depth}"
                )

            if best_score == 1.0:
                elapsed = time.perf_counter() - t_start
                return MCTSResult(
                    best_program=best_program,
                    best_score=best_score,
                    iterations=iteration + 1,
                    elapsed_sec=elapsed,
                    found_perfect=True,
                )

        elapsed = time.perf_counter() - t_start
        return MCTSResult(
            best_program=best_program,
            best_score=best_score,
            iterations=self.simulations,
            elapsed_sec=elapsed,
            found_perfect=(best_score == 1.0),
        )

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select a promising leaf using UCB1."""
        node = root
        while node.children and node.depth < self.max_depth:
            node = max(
                node.children,
                key=lambda c: c.ucb1(self.exploration_constant, node.visits),
            )
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add a new child to *node* by appending a random primitive."""
        prim_name, prim_args = self._rng.choice(_MCTS_PRIMITIVES)
        new_instr = DSLInstruction(primitive=prim_name, args=dict(prim_args))
        new_prog = DSLProgram(
            instructions=list(node.program.instructions) + [new_instr],
            source="mcts",
        )
        child = MCTSNode(program=new_prog, parent=node, depth=node.depth + 1)
        node.children.append(child)
        return child

    def _rollout(self, node: MCTSNode, pairs: List[Dict]) -> float:
        """Simulate from *node* by appending random primitives and scoring."""
        prog = node.program
        for _ in range(self.rollout_depth):
            prim_name, prim_args = self._rng.choice(_MCTS_PRIMITIVES)
            new_instr = DSLInstruction(primitive=prim_name, args=dict(prim_args))
            prog = DSLProgram(
                instructions=list(prog.instructions) + [new_instr],
                source="mcts",
            )
        return self._evaluate(prog, pairs)

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """Propagate score up to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += score
            current = current.parent

    def _evaluate(self, program: DSLProgram, pairs: List[Dict]) -> float:
        """Score a program on training pairs."""
        correct = 0
        for pair in pairs:
            result, _ = self._executor.execute(program, pair["input"])
            if result is not None and result == pair["output"]:
                correct += 1
        return correct / max(len(pairs), 1)


__all__ = ["MCTS", "MCTSResult", "MCTSNode"]
