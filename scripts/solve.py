"""
CLI entry point for solving a single ARC task.

Usage:
    python scripts/solve.py --task data/datasets/training/007bbfb7.json
    python scripts/solve.py --task 007bbfb7.json --config configs/mcts.yaml
    python scripts/solve.py --task 007bbfb7.json --algorithm beam --visualize
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

from src.api.solver_api import ARCSolver
from src.utils.config import load_config, load_default_config
from src.utils.io import load_task
from src.utils.logging_utils import setup_logger


@click.command()
@click.option("--task", "-t", required=True, help="Path to ARC task JSON file")
@click.option("--config", "-c", default=None, help="Path to YAML config file")
@click.option("--algorithm", "-a",
              type=click.Choice(["beam", "mcts", "genetic", "constraint"]),
              default=None, help="Override search algorithm")
@click.option("--time-budget", "-T", default=None, type=float,
              help="Override time budget in seconds")
@click.option("--visualize", "-v", is_flag=True, default=False,
              help="Display grid visualizations")
@click.option("--output", "-o", default=None, help="Save result JSON to this file")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              help="Logging verbosity")
def main(task, config, algorithm, time_budget, visualize, output, log_level):
    """Solve a single ARC task and print predictions."""
    setup_logger(log_level)

    # Load config
    cfg = load_config(config) if config else load_default_config()
    if algorithm:
        cfg.search.algorithm = algorithm
    if time_budget:
        cfg.solver.max_time_per_task = time_budget

    # Load task
    task_path = Path(task)
    task_id = task_path.stem
    click.echo(f"Loading task: {task_id}")
    task_dict = load_task(task_path)

    click.echo(
        f"Task: {len(task_dict['train'])} training pairs, "
        f"{len(task_dict['test'])} test inputs"
    )

    # Solve
    solver = ARCSolver(config=cfg)
    result = solver.solve(task_dict, task_id=task_id)

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"Task: {task_id}")
    click.echo(f"Total time: {result.total_elapsed_sec:.1f}s")
    click.echo(f"{'='*60}")

    for i, test_result in enumerate(result.test_results):
        click.echo(f"\nTest input [{i}]:")
        click.echo(f"  Confidence: {test_result.confidence:.3f}")
        click.echo(f"  Found perfect: {test_result.found_perfect}")
        for j, pred in enumerate(test_result.predictions, 1):
            click.echo(f"  Attempt {j}: {pred}")

    # Visualize
    if visualize:
        try:
            from src.ui.visualizer import GridVisualizer
            viz = GridVisualizer()
            from src.api.solver_api import _load_task_grids
            task_grids = _load_task_grids(task_dict)
            for i, test_pair in enumerate(task_grids.get("test", [])):
                click.echo(f"\n--- Test Input [{i}] ---")
                viz.print_grid(test_pair["input"])
                if result.test_results[i].predictions:
                    click.echo(f"--- Prediction [{i}] ---")
                    from src.core.grid.grid import ArcGrid
                    pred_grid = ArcGrid.from_list(result.test_results[i].predictions[0])
                    viz.print_grid(pred_grid)
        except Exception as e:
            click.echo(f"Visualization failed: {e}")

    # Save output
    if output:
        out_data = {
            "task_id": task_id,
            "predictions": [r.predictions for r in result.test_results],
            "total_elapsed_sec": result.total_elapsed_sec,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2)
        click.echo(f"\nResult saved to: {output}")


if __name__ == "__main__":
    main()
