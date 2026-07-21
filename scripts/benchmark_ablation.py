"""
Benchmark & Ablation Analysis Script for ARC-AGI-2026 Reasoning Engine.

Usage:
    python scripts/benchmark_ablation.py --dataset data/datasets/training --max-tasks 5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from rich.console import Console
from rich.table import Table

from src.api.solver_api import ARCSolver
from src.utils.config import load_default_config
from src.utils.io import load_task

console = Console()


@click.command()
@click.option("--dataset", "-d", default="data/datasets/training", help="Path to ARC task directory")
@click.option("--max-tasks", "-m", default=5, type=int, help="Maximum number of tasks to benchmark")
@click.option("--output", "-o", default="results/ablation_benchmark.json", help="Path to save output JSON")
def main(dataset: str, max_tasks: int, output: str) -> None:
    """Run comparative ablation benchmarks across Beam, MCTS, and Constraint search engines."""
    dataset_path = Path(dataset)
    task_files = sorted(list(dataset_path.glob("*.json")))[:max_tasks]

    if not task_files:
        console.print(f"[bold red]No task JSON files found in {dataset_path}[/bold red]")
        sys.exit(1)

    algorithms = ["beam", "mcts", "constraint"]
    results: Dict[str, Dict[str, Any]] = {algo: {"solved": 0, "total": 0, "total_time": 0.0} for algo in algorithms}

    console.print(f"[bold green]Starting Ablation Benchmark on {len(task_files)} tasks across algorithms: {algorithms}[/bold green]\n")

    for task_file in task_files:
        task_dict = load_task(task_file)
        task_id = task_file.stem

        for algo in algorithms:
            cfg = load_default_config()
            cfg.search.algorithm = algo
            cfg.solver.max_time_per_task = 3.0

            solver = ARCSolver(config=cfg)
            start_t = time.time()
            result = solver.solve(task_dict, task_id=task_id)
            elapsed = time.time() - start_t

            solved_task = result.all_correct or any(t.found_perfect for t in result.test_results)

            results[algo]["total"] += 1
            results[algo]["total_time"] += elapsed
            if solved_task:
                results[algo]["solved"] += 1

    # Render summary table
    table = Table(title="ARC-AGI-2026 Engine Ablation Benchmark Summary")
    table.add_column("Search Engine", style="cyan", no_wrap=True)
    table.add_column("Tasks Solved", style="magenta")
    table.add_column("Accuracy (%)", style="green")
    table.add_column("Avg Time / Task (s)", style="yellow")

    summary_data = []
    for algo, stats in results.items():
        total = stats["total"]
        solved = stats["solved"]
        acc = (solved / total * 100.0) if total > 0 else 0.0
        avg_time = (stats["total_time"] / total) if total > 0 else 0.0

        table.add_row(algo.upper(), f"{solved}/{total}", f"{acc:.1f}%", f"{avg_time:.2f}s")
        summary_data.append({
            "algorithm": algo,
            "solved": solved,
            "total": total,
            "accuracy": acc,
            "avg_time": avg_time,
        })

    console.print(table)

    # Save results to output JSON
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    console.print(f"\n[bold blue]Ablation benchmark results saved to {output_path}[/bold blue]")


if __name__ == "__main__":
    main()
