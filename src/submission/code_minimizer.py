"""
SubmissionMinimizer — Minimizes Python source scripts by removing docstrings, comments, typehints, and whitespace.
"""

from __future__ import annotations

import ast


class SubmissionMinimizer:
    """Minimizes Python code structure to optimize text size."""

    @staticmethod
    def minimize(code: str) -> str:
        """Strips comments and unnecessary blank lines from python source string."""
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            # Skip empty lines and comment lines
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines)
