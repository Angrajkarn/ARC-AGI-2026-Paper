"""
ARC-AGI-2 Solver
================
A research-grade modular reasoning engine for solving ARC-AGI-2 tasks.

Architecture
------------
The system implements a full symbolic + program synthesis pipeline:

    Task Loader → Grid Repr → Object Detection → Scene Graph
        → Feature Extraction → Rule Discovery → DSL Program Gen
        → Program Search → Execution → Verification
        → Reflection → Ranking → Submission

Quick Start
-----------
    # Install
    pip install -e ".[dev]"

    # Download ARC data
    python scripts/download_data.py

    # Solve tasks
    python scripts/solve.py --task data/datasets/training/007bbfb7.json

    # Evaluate on training set
    python scripts/evaluate.py --dataset data/datasets/training --max-tasks 100

    # Generate submission
    python scripts/submit.py --dataset data/datasets/evaluation --output submission.json

Repository Structure
--------------------
    src/core/          - Grid, Object, SceneGraph data structures
    src/dsl/           - Domain-specific language (primitives + executor)
    src/vision/        - Geometry, topology, color, feature extraction
    src/reasoning/     - Rule discovery, planning, program synthesis
    src/search/        - Beam search, MCTS, genetic, constraint solver
    src/verifier/      - Exact match + consistency verification
    src/reflection/    - Failure analysis and retry logic
    src/memory/        - Persistent program library
    src/ensemble/      - Multi-solver ensemble
    src/ranking/       - Candidate scoring and ranking
    src/llm/           - LLM reasoning assistant (Ollama adapter)
    src/evaluation/    - Local evaluator and benchmark runner
    src/submission/    - submission.json generator
    src/api/           - Clean Python API
    src/ui/            - Grid visualizer
    configs/           - YAML configurations
    scripts/           - CLI entry points
    tests/             - Unit + integration tests

Configuration
-------------
All behaviour is controlled via YAML config files in `configs/`.
Override specific keys from the CLI:

    python scripts/solve.py --config configs/mcts.yaml

Requirements
------------
Python 3.10+. See pyproject.toml for dependencies.
No internet connection required during inference (Kaggle-compatible).

License
-------
MIT
"""
