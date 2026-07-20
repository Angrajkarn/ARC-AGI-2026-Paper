"""
Parallel Dataset Evaluator CLI — run fast multi-core benchmark.

Usage:
    python scripts/evaluate_parallel.py --dataset data/datasets/training --max-tasks 50 --workers 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

from src.evaluation.parallel_evaluator import ParallelEvaluator
from src.utils.config import load_config, load_default_config
from src.utils.io import load_tasks_from_dir
from src.utils.logging_utils import setup_logger


@click.command()
@click.option("--dataset", "-d", required=True, help="Path to dataset directory")
@click.option("--max-tasks", "-n", default=None, type=int, help="Limit number of tasks")
@click.option("--workers", "-w", default=4, type=int, help="Number of CPU worker processes")
@click.option("--config", "-c", default=None, help="Path to YAML config file")
@click.option("--output", "-o", default="results/evaluation_parallel.json", help="Output JSON path")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(dataset, max_tasks, workers, config, output, log_level):
    """Run parallel evaluation across multiple worker processes."""
    setup_logger(log_level)

    cfg = load_config(config) if config else load_default_config()

    click.echo(f"Loading tasks from: {dataset}")
    tasks = load_tasks_from_dir(dataset, max_tasks=max_tasks)
    click.echo(f"Loaded {len(tasks)} tasks")

    evaluator = ParallelEvaluator(config=cfg, max_workers=workers)
    summary = evaluator.evaluate_dataset_parallel(tasks, max_tasks=max_tasks)

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo(f"PARALLEL EVALUATION SUMMARY ({workers} Workers)")
    click.echo(f"{'='*60}")
    click.echo(f"Total tasks:     {summary.total_tasks}")
    click.echo(f"Solved tasks:    {summary.solved_tasks}")
    click.echo(f"Solve rate:      {summary.solve_rate:.1%}")
    click.echo(f"Avg time/task:   {summary.avg_elapsed_sec:.1f}s")
    click.echo(f"{'='*60}")

    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result_data = {
        "summary": {
            "total_tasks": summary.total_tasks,
            "solved_tasks": summary.solved_tasks,
            "solve_rate": summary.solve_rate,
            "avg_elapsed_sec": summary.avg_elapsed_sec,
            "workers": workers,
        },
        "per_task": [
            {
                "task_id": r.task_id,
                "solve_rate": r.solve_rate,
                "num_correct": r.num_correct,
                "num_tests": r.num_tests,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in summary.task_results
        ],
    }
    with open(output, "w") as f:
        json.dump(result_data, f, indent=2)
    click.echo(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
