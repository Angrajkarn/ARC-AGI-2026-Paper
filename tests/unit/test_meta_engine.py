"""
Unit tests for MetaLearningEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.meta_learning.meta_engine import MetaLearningEngine


class TestMetaLearningEngine:
    def test_extract_task_features(self):
        inp = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        out = ArcGrid(pixels=np.array([[1, 2, 5], [3, 4, 0]]), background=0)
        pairs = [{"input": inp, "output": out}]

        features = MetaLearningEngine.extract_task_features(pairs)
        assert features["max_size"] == 6  # output grid size is 2x3=6
        assert features["unique_colors"] == 6  # colors 0, 1, 2, 3, 4, 5
        assert features["size_changes"] is True

    def test_recommend_config_standard(self):
        inp = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        out = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        pairs = [{"input": inp, "output": out}]

        engine = MetaLearningEngine()
        config = engine.recommend_config(pairs)

        # Standard grid sizes should get standard configs
        assert config["beam_width"] == 10
        assert config["max_depth"] == 6
        assert config["time_budget"] == 30.0
