"""
DSL Primitives initialization.
"""

from src.dsl.primitives import transforms
from src.dsl.primitives import subgrid_primitives
from src.dsl.primitives import combinators
from src.dsl.primitives import advanced_primitives
from src.dsl.primitives import higher_order

from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY

__all__ = [
    "transforms",
    "subgrid_primitives",
    "combinators",
    "advanced_primitives",
    "higher_order",
    "PRIMITIVE_REGISTRY",
]
