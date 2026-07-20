"""
ArcGrid — core grid data structure for ARC-AGI-2 tasks.

Grids are the fundamental unit of the ARC format.  Rather than treating
them as raw NumPy tensors we wrap them in a rich dataclass that carries
derived metadata (size, color set, background) and convenience methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np


# ARC colour palette: integers 0-9 map to named colours for readability.
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}


@dataclass
class ArcGrid:
    """Rich representation of a single ARC grid.

    Attributes:
        pixels:     2-D NumPy array of shape (height, width), dtype int8, values 0-9.
        background: The most-common (or explicitly set) background colour.
        metadata:   Arbitrary key→value store for task-level annotations.
    """

    pixels: np.ndarray  # shape (H, W), dtype int8
    background: int = 0
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties (computed once on first access via property)
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def colors(self) -> Set[int]:
        """Set of unique colour values present in the grid."""
        return set(np.unique(self.pixels).tolist())

    @property
    def non_background_colors(self) -> Set[int]:
        return self.colors - {self.background}

    @property
    def non_background_ratio(self) -> float:
        """Fraction of pixels that are not background."""
        non_bg = int((self.pixels != self.background).sum())
        return non_bg / max(self.num_pixels, 1)

    @property
    def size(self) -> Tuple[int, int]:
        """(height, width) tuple."""
        return (self.height, self.width)

    @property
    def num_pixels(self) -> int:
        return self.height * self.width

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        grid: List[List[int]],
        background: Optional[int] = None,
    ) -> "ArcGrid":
        """Create an ArcGrid from a 2-D Python list (ARC JSON format).

        Args:
            grid:       2-D list of ints 0-9.
            background: Override background colour detection.

        Returns:
            ArcGrid instance.
        """
        arr = np.array(grid, dtype=np.int8)
        if background is None:
            # Detect background as most-frequent colour
            values, counts = np.unique(arr, return_counts=True)
            background = int(values[np.argmax(counts)])
        return cls(pixels=arr, background=background)

    @classmethod
    def empty(cls, height: int, width: int, fill: int = 0) -> "ArcGrid":
        """Create a blank grid filled with *fill* colour."""
        return cls(pixels=np.full((height, width), fill, dtype=np.int8), background=fill)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, background: int = 0) -> "ArcGrid":
        """Wrap a NumPy array as an ArcGrid."""
        return cls(pixels=arr.astype(np.int8), background=background)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_list(self) -> List[List[int]]:
        """Convert back to ARC JSON format (2-D list of ints)."""
        return self.pixels.tolist()

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.pixels.copy()

    # ------------------------------------------------------------------
    # Equality & hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArcGrid):
            return NotImplemented
        return (
            self.pixels.shape == other.pixels.shape
            and bool(np.array_equal(self.pixels, other.pixels))
        )

    def __hash__(self) -> int:
        return hash(self.pixels.tobytes())

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "ArcGrid":
        """Return a deep copy of the grid."""
        return ArcGrid(
            pixels=self.pixels.copy(),
            background=self.background,
            metadata=self.metadata.copy(),
        )

    # ------------------------------------------------------------------
    # Pixel access helpers
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> int:
        """Get the colour at (row, col)."""
        return int(self.pixels[row, col])

    def set(self, row: int, col: int, color: int) -> None:
        """Set the colour at (row, col) in-place."""
        self.pixels[row, col] = color

    def in_bounds(self, row: int, col: int) -> bool:
        """Check whether (row, col) is within the grid."""
        return 0 <= row < self.height and 0 <= col < self.width

    def pixels_of_color(self, color: int) -> List[Tuple[int, int]]:
        """Return all (row, col) positions with the given colour."""
        rows, cols = np.where(self.pixels == color)
        return list(zip(rows.tolist(), cols.tolist()))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ArcGrid(height={self.height}, width={self.width}, "
            f"colors={sorted(self.colors)}, background={self.background})"
        )

    def __str__(self) -> str:
        rows = []
        for row in self.pixels.tolist():
            rows.append(" ".join(str(c) for c in row))
        return "\n".join(rows)


__all__ = ["ArcGrid", "COLOR_NAMES"]
