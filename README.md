# ARC-AGI-2026 Reasoning Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Status](https://img.shields.io/badge/tests-208%20passed-green.svg)](#-verification--testing)
[![Architecture](https://img.shields.io/badge/architecture-Neuro--Symbolic--Topology-purple.svg)](#-architecture-overview)

An advanced, state-of-the-art neuro-symbolic reasoning engine designed to solve the Abstraction and Reasoning Corpus (ARC-AGI) visual abstraction tasks. By combining structured search heuristics with multi-engine program synthesis, the solver constructs domain-specific language (DSL) programs matching object transformation laws across demonstrations.

---

## 🚀 Core Paradigm & Key Features

### 1. Neuro-Symbolic Program Synthesis
- **Multi-Engine Search**: Parallel execution of **Monte Carlo Tree Search (MCTS)**, **Beam Search**, and **Genetic Algorithms** over AST structures.
- **Relational Scene Graphs**: Converts grids into object-oriented graph representations modeling directional, enclosure, and contact interfaces.
- **Continuous Relaxed Optimization**: Relaxes discrete program candidate search spaces into continuous parameters for differentiable search trials.

### 2. Topological & Structural Invariants
- **Homology Loop Verification**: Extracts Betti numbers ($\beta_0$, $\beta_1$) and Euler Characteristic signatures to prune candidate programs violating grid topology.
- **Equivariance-Preserving Spatial Encoder**: Computes shape signatures invariant under Dihedral $D_4$ rotations and reflection symmetries.
- **Multi-Edge Hypergraph Grammar**: Models alignment configurations and color groups across demonstration grid transitions.

### 3. Verification & Search Enhancements
- **Dynamic Task Curriculum Scheduler**: Orders training demonstration pairs from easiest to hardest based on Shannon entropy, area metrics, and scale/color variance factors to warm-start searches.
- **Topology-Preserving Skeletonization**: Iteratively thins pixel clusters into 1-pixel medial axes for path/maze routing verification.
- **Self-Attention Transformer Priors**: Ranks optimal DSL transformations utilizing scaled dot-product key-query attention maps over grid shape attributes.

---

## 🧬 Architectural Deep-Dive (Phase 18–22 Advanced Modules)

```
                               ┌────────────────────────┐
                               │   Input/Output Grids   │
                               └───────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Object Parser & Graph  │
                              │        Clustering       │
                              └────────────┬────────────┘
                                           │
                     ┌─────────────────────▼─────────────────────┐
                     │          Topological Prior Selector       │
                     │  (Contour Tracing, Persistence Homology)  │
                     └─────────────────────┬─────────────────────┘
                                           │
                     ┌─────────────────────▼─────────────────────┐
                     │       Multi-Engine Program Synthesis      │
                     │  (MCTS, Dependency Priors, Trace Cache)   │
                     └─────────────────────┬─────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │ Consensus Voting & TTA  │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Sanitized Prediction   │
                              └─────────────────────────┘
```

### 🔍 Scene Graph Object Clustering
- Identifies equivalences across segmented scene graph objects by clustering spatial bounding features.
- Employs exact bounding area matrices to group objects under rotational or mirror transformations.

### 🛠️ Self-Correction Refinement Discriminator
- Analyzes predicted output grids for high-frequency noise or invalid pixel boundaries.
- Scores output sanity to trigger backtracking during code generation when anomalous single-pixel structures are found.

### 📉 Contrastive Program Code Reducer
- Minimizes program complexity by simplifying redundant primitive operations (e.g., matching sequential horizontal flips or double $180^\circ$ rotations) into identity transforms.

### 🔄 Symmetric Grid Pattern Autocomplete Prior
- Identifies bilateral reflection axes (horizontal and vertical) across demo shapes.
- Automatically autocompletes empty grid values using symmetrical counterparts.

### 🗺️ Equivariant Coordinate Grid Prior
- Maps index positions $(r, c)$ using reflection or translation transformation matrices to verify coordinate validity under D4 actions.

### 🔗 Dynamic DSL Program Dependency Prior Selector
- Evaluates transition sequences of DSL instructions (e.g. `crop` -> `scale_2x`) and assigns prior probability weights based on logical order execution rules.

### 🎨 Global Color Palette Transfer Prior
- Discovers frequency-ranked color mappings between source and destination grids to parameterize color-swap operations.

### 🕸️ Program Graph Representation Embedder
- Represents the complete DSL instruction sequence as a directed execution flow graph where operations are nodes and dependencies are edges.

### ⚖️ Dynamic Task Scaling Prior Selector
- Predicts target grid dimensions by computing the scaling multiplier ratio between input and output demonstration grids.

### 🖌️ Object Connectivity Contour Tracing Prior
- Traces outer boundary/perimeter coordinates of segmented object shapes using 8-way connectivity boundary detection.

### 📏 Contrastive Program Distance Metric Evaluator
- Computes Levenshtein-style edit distance on program operation sequences to cluster related programs.

### 👁️ Hierarchical Sub-grid Pattern Attention Prior
- Extracts local 3x3 patches and computes cosine similarity scores across patch vectors to measure spatial regularity.

### 💾 Dynamic Program Execution Trace Log Cacher
- Caches grid hashes generated at intermediate program synthesis execution steps to prune redundant sub-trees and cycle loops.

### 📐 Contrastive Color Topology Prior Selector
- Compares topological shapes and boundary counts of colored pixels between source and output to enforce consistency.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Angrajkarn/ARC-AGI-2026-Paper.git
   cd ARC-AGI-2026-Paper
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage & Commands

### Run Unit Tests
Validate all 208 tests in the test suite:
```powershell
pytest
```

### View Performance Statistics Charts
Render ASCII performance benchmarks comparing search algorithms:
```powershell
python scripts/plot_performance.py
```

### Build & Verify Kaggle Release
Compile the standalone submission notebook and verify execution in an offline sandbox:
```powershell
python scripts/build_kaggle_release.py
```
```powershell
python scripts/verify_submission.py
```

### Launch Interactive Web Dashboard
Run the Streamlit visualization inspector:
```powershell
streamlit run src/ui/app.py
```
