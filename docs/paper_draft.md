# ARC-AGI-2026: A Hybrid Symbolic Scene-Graph and Neural-Guided Program Synthesis Engine for Abstraction and Reasoning

**Authors**: ARC Research Team  
**Date**: July 2026  
**Repository**: [ARC-AGI-2026-Paper](https://github.com/Angrajkarn/ARC-AGI-2026-Paper)

---

## Abstract

The Abstraction and Reasoning Corpus (ARC-AGI) challenges artificial intelligence systems to infer generalizable visual-spatial rules from extremely small sample sets (typically 2–4 demonstrations). In this paper, we present **ARC-AGI-2026**, a modular reasoning engine combining object-centric scene graphs, rich mathematical DSL primitives, multi-dimensional similarity heuristics, and neural prior search. We formalize grid spatial relations into a directed multigraph representation and evaluate performance across Beam Search, Monte Carlo Tree Search (MCTS), and Constraint Solving.

---

## 1. Introduction

ARC-AGI requires fluid intelligence: abstracting objects, spatial geometry, color invariants, and relational transformations without massive pre-training dataset reliance. Standard end-to-end deep neural networks often struggle with exact grid discrete transformations and general boundary extrapolation. 

Our system addresses these limitations by pairing:
1. **Object-Centric Scene & Spatial Graphs**: Extracting connected components, bounding box hierarchies, and directional containment topology.
2. **Expressive Domain-Specific Language (DSL)**: Incorporating morphological operations (dilation, erosion, open/close), pattern tiling, cellular automata steps, and functional combinators.
3. **Multi-Engine Search**: MCTS and Beam search guided by composite IoU and structural entropy similarity heuristics.

---

## 2. Mathematical & Architectural Formalism

### 2.1 Spatial Relational Graph

Let an ARC grid be represented as a 2D discrete lattice $G \in \{0, \dots, 9\}^{H \times W}$.
Objects $\mathcal{O} = \{O_1, O_2, \dots, O_k\}$ are extracted using 4- or 8-connectivity component detection.

We construct a Spatial Relational Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where each node $v_i \in \mathcal{V}$ corresponds to object $O_i$, and directed edges $e_{ij} \in \mathcal{E}$ carry relation labels:

$$\text{Rel}(O_i, O_j) \subseteq \{\text{ABOVE}, \text{BELOW}, \text{LEFT\_OF}, \text{RIGHT\_OF}, \text{ENCLOSES}, \text{INSIDE}, \text{TOUCHING}, \text{ALIGNED}\}$$

Enclosure $\text{ENCLOSES}(O_i, O_j)$ is computed via bounding box containment and exterior flood-fill connectivity:

$$\text{BBox}(O_j) \subseteq \text{BBox}(O_i) \quad \land \quad \text{Path}_{\text{ext}}(O_j) = \emptyset$$

### 2.2 Domain-Specific Language (DSL)

Programs $P$ are compositions of primitive operations $f_i \in \mathcal{F}$:

$$P = f_n \circ f_{n-1} \circ \dots \circ f_1(G)$$

Primitive categories include:
- **Spatial Transforms**: `rotate_90`, `mirror_horizontal`, `mirror_vertical`, `translate`
- **Morphology & Boundaries**: `dilate`, `erode`, `extract_outline`, `fill_enclosed_holes`
- **Subgrid & Pattern Actions**: `crop_to_content`, `pattern_repeat`, `split_by_grid_lines`
- **Cellular Automata**: `step_ca_majority`

---

## 3. Neural-Guided Search & Multi-Dimensional Heuristic

### 3.1 Composite Heuristic Evaluation

Candidate program outputs $\hat{G}$ are evaluated against ground truth target $G^*$ using a continuous gradient score $S(\hat{G}, G^*) \in [0.0, 1.0]$:

$$S(\hat{G}, G^*) = w_1 \cdot \text{IoU}(\hat{G}, G^*) + w_2 \cdot \text{HistSim}(\hat{G}, G^*) + w_3 \cdot \Delta \text{Entropy}(\hat{G}, G^*) + w_4 \cdot \text{DimSim}(\hat{G}, G^*)$$

Where:
- $\text{IoU}(\hat{G}, G^*)$ measures spatial pixel overlap of non-background regions.
- $\text{HistSim}(\hat{G}, G^*)$ computes normalized color histogram intersection.
- $\Delta \text{Entropy}$ measures structural entropy alignment.

### 3.2 UCT Expansion with Neural Priors

In Monte Carlo Tree Search (MCTS), node selection follows the Upper Confidence Bound for Trees (UCT) augmented with primitive prior probabilities $P(s, a)$:

$$UCT(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

Where $P(s, a)$ is predicted by `NeuralSearchPrior` based on input-output dimension ratios and color transition matrices.

---

## 4. Benchmark & Ablation Protocol

The engine provides an automated ablation protocol (`scripts/benchmark_ablation.py`) to systematically compare search algorithms (Beam Search vs. MCTS vs. Constraint Solving) across:
- **Task Accuracy**: Percentage of exact grid matches on test pairs.
- **Search Efficiency**: Total expanded nodes and time budget usage per task.
- **Primitive Coverage**: Solvability with morphology vs. pure geometric transforms.

---

## 5. Conclusion

The **ARC-AGI-2026** engine demonstrates that combining explicit object-spatial relational graphs with continuous multi-dimensional heuristics and modular DSL synthesis provides a powerful framework for solving complex spatial reasoning tasks. Future directions include integrating transformer-based neuro-symbolic program predictors for zero-shot primitive selection.
