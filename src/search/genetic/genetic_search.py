"""
Genetic/Evolutionary Search for ARC DSL program synthesis.

Algorithm:
  1. Initialise population from hypotheses + random programs.
  2. Evaluate fitness = train_accuracy + simplicity bonus.
  3. Select parents via tournament selection.
  4. Crossover: combine instruction sequences from two parents.
  5. Mutate: randomly add, remove, or replace instructions.
  6. Elitism: keep top-k individuals unchanged.
  7. Repeat for N generations.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Gene pool: (name, args) pairs
_GENE_POOL = [
    ("mirror_horizontal", {}),
    ("mirror_vertical", {}),
    ("mirror_diagonal", {}),
    ("rotate_90", {"times": 1}),
    ("rotate_90", {"times": 2}),
    ("rotate_90", {"times": 3}),
    ("crop_to_content", {}),
    ("fill_holes", {}),
    ("hollow", {}),
    ("normalize_colors", {}),
    ("gravity", {"direction": "down"}),
    ("gravity", {"direction": "up"}),
    ("gravity", {"direction": "left"}),
    ("gravity", {"direction": "right"}),
] + [
    ("replace_color", {"source_color": s, "target_color": t})
    for s in range(10) for t in range(10) if s != t
] + [
    ("swap_colors", {"color_a": a, "color_b": b})
    for a in range(5) for b in range(a + 1, 5)
] + [
    ("add_border", {"color": c, "thickness": 1}) for c in range(5)
]


@dataclass
class Individual:
    """A member of the genetic population."""

    program: DSLProgram
    fitness: float = 0.0
    train_accuracy: float = 0.0


@dataclass
class GeneticResult:
    best_program: Optional[DSLProgram]
    best_score: float
    generation: int
    elapsed_sec: float
    found_perfect: bool


class GeneticSearch:
    """Evolutionary search for ARC DSL programs.

    Usage::

        search = GeneticSearch(population_size=50, generations=30)
        result = search.search(hypotheses, pairs, time_budget=120)
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elitism: int = 5,
        max_program_length: int = 8,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.max_program_length = max_program_length
        self._executor = DSLExecutor(debug=False)
        self._rng = random.Random(seed)

    def search(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 120.0,
    ) -> GeneticResult:
        """Run genetic search.

        Args:
            hypotheses: Seed individuals from rule discovery.
            pairs:      Training pairs.
            time_budget: Wall-clock limit.

        Returns:
            GeneticResult with best program found.
        """
        t_start = time.perf_counter()
        population = self._init_population(hypotheses, pairs)
        best = max(population, key=lambda i: i.fitness) if population else None
        generation = 0

        for generation in range(self.generations):
            if time.perf_counter() - t_start > time_budget:
                break

            # Elitism: carry forward top individuals
            population.sort(key=lambda i: i.fitness, reverse=True)
            new_pop: List[Individual] = population[: self.elitism]

            # Fill rest with crossover + mutation
            while len(new_pop) < self.population_size:
                if self._rng.random() < self.crossover_rate and len(population) >= 2:
                    p1, p2 = self._tournament_select(population, k=3), self._tournament_select(population, k=3)
                    child_prog = self._crossover(p1.program, p2.program)
                else:
                    parent = self._tournament_select(population, k=3)
                    child_prog = copy.deepcopy(parent.program)

                child_prog = self._mutate(child_prog)
                child = Individual(program=child_prog)
                child.train_accuracy, child.fitness = self._evaluate(child_prog, pairs)
                new_pop.append(child)

                if child.fitness == 1.0:
                    elapsed = time.perf_counter() - t_start
                    logger.info(f"Genetic: perfect individual found at gen {generation}")
                    return GeneticResult(
                        best_program=child_prog,
                        best_score=child.fitness,
                        generation=generation,
                        elapsed_sec=elapsed,
                        found_perfect=True,
                    )

            population = new_pop
            gen_best = max(population, key=lambda i: i.fitness)
            if best is None or gen_best.fitness > best.fitness:
                best = gen_best
                logger.debug(
                    f"Gen {generation}: best_fitness={best.fitness:.3f}, "
                    f"acc={best.train_accuracy:.2f}, "
                    f"len={best.program.length}"
                )

        elapsed = time.perf_counter() - t_start
        return GeneticResult(
            best_program=best.program if best else None,
            best_score=best.fitness if best else 0.0,
            generation=generation,
            elapsed_sec=elapsed,
            found_perfect=(best.fitness == 1.0 if best else False),
        )

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _init_population(
        self, hypotheses: List[Hypothesis], pairs: List[Dict]
    ) -> List[Individual]:
        population: List[Individual] = []

        # Seed from hypotheses
        for h in hypotheses:
            ind = Individual(program=copy.deepcopy(h.candidate_program))
            ind.train_accuracy, ind.fitness = self._evaluate(ind.program, pairs)
            population.append(ind)

        # Fill with random individuals
        while len(population) < self.population_size:
            length = self._rng.randint(1, self.max_program_length)
            instrs = [
                DSLInstruction(primitive=n, args=dict(a))
                for n, a in self._rng.choices(_GENE_POOL, k=length)
            ]
            prog = DSLProgram(instructions=instrs, source="genetic")
            ind = Individual(program=prog)
            ind.train_accuracy, ind.fitness = self._evaluate(prog, pairs)
            population.append(ind)

        return population

    def _tournament_select(
        self, population: List[Individual], k: int = 3
    ) -> Individual:
        """Tournament selection of size k."""
        candidates = self._rng.sample(population, min(k, len(population)))
        return max(candidates, key=lambda i: i.fitness)

    def _crossover(self, prog_a: DSLProgram, prog_b: DSLProgram) -> DSLProgram:
        """Single-point crossover of two instruction sequences."""
        a = list(prog_a.instructions)
        b = list(prog_b.instructions)
        if not a or not b:
            return DSLProgram(instructions=list(a or b), source="genetic")
        cut_a = self._rng.randint(0, len(a))
        cut_b = self._rng.randint(0, len(b))
        new_instrs = a[:cut_a] + b[cut_b:]
        new_instrs = new_instrs[: self.max_program_length]
        return DSLProgram(instructions=new_instrs, source="genetic")

    def _mutate(self, prog: DSLProgram) -> DSLProgram:
        """Apply random mutations to a program."""
        instrs = list(prog.instructions)

        # Add
        if self._rng.random() < self.mutation_rate and len(instrs) < self.max_program_length:
            name, args = self._rng.choice(_GENE_POOL)
            pos = self._rng.randint(0, len(instrs))
            instrs.insert(pos, DSLInstruction(primitive=name, args=dict(args)))

        # Remove
        if self._rng.random() < self.mutation_rate and len(instrs) > 1:
            idx = self._rng.randint(0, len(instrs) - 1)
            instrs.pop(idx)

        # Replace
        if self._rng.random() < self.mutation_rate and instrs:
            idx = self._rng.randint(0, len(instrs) - 1)
            name, args = self._rng.choice(_GENE_POOL)
            instrs[idx] = DSLInstruction(primitive=name, args=dict(args))

        return DSLProgram(instructions=instrs, source="genetic")

    def _evaluate(self, program: DSLProgram, pairs: List[Dict]) -> Tuple[float, float]:
        """Return (train_accuracy, fitness)."""
        correct = 0
        for pair in pairs:
            result, _ = self._executor.execute(program, pair["input"])
            if result is not None and result == pair["output"]:
                correct += 1
        acc = correct / max(len(pairs), 1)
        simplicity = max(0.0, 1.0 - program.length / self.max_program_length)
        fitness = 0.7 * acc + 0.3 * simplicity
        return acc, fitness


__all__ = ["GeneticSearch", "GeneticResult", "Individual"]
