"""
RetrospectionDatabase — stores successfully solved program templates in a local JSON registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class RetrospectionDatabase:
    """Registry to save and load discovered program templates to seed search processes."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Path("data/memory/retrospection_db.json")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, List[str]] = self._load()

    def _load(self) -> Dict[str, List[str]]:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_template(self, category: str, program_str: str) -> None:
        """Stores program template in database under a given category."""
        if category not in self.templates:
            self.templates[category] = []
        if program_str not in self.templates[category]:
            self.templates[category].append(program_str)
        self._commit()

    def get_templates(self, category: str) -> List[str]:
        """Retrieves program templates for a category."""
        return self.templates.get(category, [])

    def _commit(self) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.templates, f, indent=2)
