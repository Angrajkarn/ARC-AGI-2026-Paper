"""
Unit tests for KaggleExporter.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.submission.kaggle_exporter import KaggleExporter


class TestKaggleExporter:
    def test_export_creates_file(self, tmp_path: Path):
        exporter = KaggleExporter()
        out_file = tmp_path / "submission_bundle.py"

        result_path = exporter.export(out_file)
        assert result_path.exists()
        assert result_path.stat().st_size > 1000

        content = result_path.read_text(encoding="utf-8")
        assert "Auto-generated Kaggle Submission Script" in content
        assert "ArcGrid" in content
