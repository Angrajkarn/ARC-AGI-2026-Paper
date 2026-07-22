"""
Unit tests for RetrospectionDatabase.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.memory.retrospection_db import RetrospectionDatabase


class TestRetrospectionDb:
    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "retrospection_test.json"
            db = RetrospectionDatabase(db_path=db_path)

            assert len(db.get_templates("alignment")) == 0

            db.save_template("alignment", "mirror_horizontal -> rotate_90")
            assert "mirror_horizontal -> rotate_90" in db.get_templates("alignment")

            # Reload to test persistence
            db2 = RetrospectionDatabase(db_path=db_path)
            assert "mirror_horizontal -> rotate_90" in db2.get_templates("alignment")
