"""
ASTGeneticOperators — Sub-tree crossover and argument mutation operators for Evolutionary Search.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class ASTGeneticOperators:
    """Genetic operators for DSL AST program populations."""

    @staticmethod
    def crossover(parent1: DSLProgram, parent2: DSLProgram) -> Tuple[DSLProgram, DSLProgram]:
        """Performs single-point crossover over instruction lists."""
        insts1 = [copy.deepcopy(ins) for ins in parent1.instructions]
        insts2 = [copy.deepcopy(ins) for ins in parent2.instructions]

        if not insts1 or not insts2:
            return parent1, parent2

        pt1 = random.randint(0, len(insts1))
        pt2 = random.randint(0, len(insts2))

        child1_insts = insts1[:pt1] + insts2[pt2:]
        child2_insts = insts2[:pt2] + insts1[pt1:]

        return DSLProgram(instructions=child1_insts), DSLProgram(instructions=child2_insts)

    @staticmethod
    def mutate(program: DSLProgram, mutation_rate: float = 0.3) -> DSLProgram:
        """Mutates primitive arguments or inserts/deletes operations in DSLProgram."""
        insts = [copy.deepcopy(ins) for ins in program.instructions]

        if not insts or random.random() < mutation_rate:
            # Insert random primitive
            all_prims = list(PRIMITIVE_REGISTRY.keys())
            if all_prims:
                name = random.choice(all_prims)
                insts.append(DSLInstruction(primitive=name, args={}))

        if len(insts) > 1 and random.random() < mutation_rate:
            # Delete random instruction
            idx = random.randint(0, len(insts) - 1)
            insts.pop(idx)

        # Argument mutation
        for ins in insts:
            if random.random() < mutation_rate:
                if "color" in ins.args:
                    ins.args["color"] = random.randint(0, 9)
                elif "times" in ins.args:
                    ins.args["times"] = random.randint(1, 3)

        return DSLProgram(instructions=insts)
