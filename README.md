# ARC-AGI-2026 Paper & Reasoning Engine

A research-grade modular reasoning engine for solving **ARC-AGI-2 (Abstraction and Reasoning Corpus)** tasks, incorporating symbolic graph representations, Domain-Specific Language (DSL) program synthesis, multi-engine search, and LLM-assisted reflection.

---

## 🌟 Key Features

- **Symbolic & Object Representation**: Converts ARC grids into rich object graphs, scene graphs, and topology/color feature representations.
- **Domain-Specific Language (DSL)**: Comprehensive primitive operations for spatial transformations, object manipulations, subgrid actions, and functional combinators.
- **Multi-Engine Search**: Supports **Beam Search**, **Monte Carlo Tree Search (MCTS)**, **Genetic Search**, and **Constraint Solving**.
- **Self-Reflection & Verification**: Program candidate scoring, consistency verification, failure reflection, and memory library persistence.
- **Kaggle & Benchmark Compatible**: Offline execution support and automated bundle exporter (`scripts/export_kaggle_bundle.py`) for Kaggle notebook submissions.

---

## 🏗️ Architecture Overview

The solver processes tasks through an automated multi-stage reasoning pipeline:

```text
  Task Loader → Grid Representation → Object & Scene Graph Extraction
        ↓
  Feature Extraction & Rule Discovery → DSL Program Synthesis
        ↓
  Multi-Engine Search (Beam / MCTS / Genetic / Constraint)
        ↓
  Program Execution & Verifier → Reflection & Candidate Ranking
        ↓
  Ensemble Aggregation → Final Submission Generator
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/Angrajkarn/ARC-AGI-2026-Paper.git
cd ARC-AGI-2026-Paper
pip install -e ".[dev]"
```

### 2. Download ARC Datasets

Fetch the official ARC-AGI dataset files:

```bash
python scripts/download_data.py
```

### 3. Solve a Single Task

Run the solver on a specific task JSON:

```bash
python scripts/solve.py --task data/datasets/training/007bbfb7.json --algorithm mcts --visualize
```

### 4. Evaluate Performance

Run local evaluation on the dataset:

```bash
# Evaluate on training tasks
python scripts/evaluate.py --dataset data/datasets/training --max-tasks 50

# Run parallel multi-core evaluation
python scripts/evaluate_parallel.py --dataset data/datasets/training --workers 4
```

### 5. Export Kaggle Submission Bundle

Pack the codebase and dependencies into a self-contained notebook bundle for submission:

```bash
python scripts/export_kaggle_bundle.py --output results/kaggle_submission_notebook.py
```

---

## 📁 Repository Structure

```text
ARC-AGI-2026-Paper/
├── src/
│   ├── core/          # ArcGrid, ArcObject, and SceneGraph data structures
│   ├── dsl/           # Domain-Specific Language primitives, AST, and executor
│   ├── vision/        # Geometry, topology, color, and feature extraction
│   ├── reasoning/     # Rule discovery, task planner, and shape predictor
│   ├── search/        # Search algorithms (Beam, MCTS, Genetic, Constraint)
│   ├── verifier/      # Exact match & candidate consistency verification
│   ├── reflection/    # Reflector logic for failure analysis & retries
│   ├── memory/        # Program library and memory store
│   ├── ensemble/      # Multi-solver ensemble module
│   ├── ranking/       # Candidate score ranker
│   ├── llm/           # LLM reasoning assistant adapter
│   ├── evaluation/    # Local benchmark evaluator
│   ├── submission/    # Submission generator (submission.json)
│   ├── api/           # Python API interface
│   └── ui/            # Grid visualizer
├── configs/           # YAML solver configurations
├── scripts/           # CLI entry points
├── tests/             # Unit and integration test suite
├── pyproject.toml     # Package configuration & dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Configuration

Engine hyperparameters are defined in YAML files located in `configs/`. You can specify custom configurations via the CLI:

```bash
python scripts/solve.py --task data/datasets/training/007bbfb7.json --config configs/mcts.yaml
```

---

## 📄 License

Distributed under the **Apache License 2.0**. See `LICENSE` for details.
